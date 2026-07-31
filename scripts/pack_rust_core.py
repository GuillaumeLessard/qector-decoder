#!/usr/bin/env python3
"""Pack / unpack the proprietary Rust core for CI secret injection.

Why this exists
---------------
`.gitignore` carries `src/*` with a single `!src/lib.rs` exception, so the
public repository contains exactly **one** tracked `.rs` file. The other ~45
files — the entire decoder core — never reach GitHub. CI therefore cannot build
a wheel from a fresh checkout unless the core is restored first, which is what
the `RUST_SRC_B64_1/2/3` Actions secrets are for.

Anything changed in `src/*.rs` on a developer machine is **local-only** until
this script is re-run and the secrets are refreshed. That is the single most
common way a real fix silently fails to ship.

Usage
-----
Pack the current tree into secret-sized chunks::

    python scripts/pack_rust_core.py pack --out .secrets/

Verify a pack round-trips byte-identically (does not write source)::

    python scripts/pack_rust_core.py verify --in .secrets/

Restore into a checkout (what CI runs)::

    python scripts/pack_rust_core.py unpack --in .secrets/

In CI the chunks arrive as environment variables rather than files::

    RUST_SRC_B64_1=... RUST_SRC_B64_2=... RUST_SRC_B64_3=... \\
        python scripts/pack_rust_core.py unpack --from-env

Format
------
`tar` of the tracked-but-ignored Rust sources → gzip (deterministic: fixed
mtime, sorted names) → base64 → split into `NUM_CHUNKS` equal parts. Chunk *i*
goes into secret `RUST_SRC_B64_{i}`. Concatenating the chunks in order and
base64-decoding reproduces the archive exactly.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import lzma
import os
import pathlib
import sys
import tarfile

# GitHub Actions caps a single secret at 48 KB. The historical layout was three
# chunks (`RUST_SRC_B64_1/2/3`), which fit when the core was smaller; the core
# has since grown past that, so the chunk count is now derived from the payload
# and 3 is only a floor. Chunk N lives in secret `RUST_SRC_B64_N`.
# Two ceilings apply, and the tighter one wins:
#   * GitHub Actions secret:      48 KB
#   * Windows environment variable: 32,767 chars — and `unpack --from-env` has to
#     work on the maintainer's Windows box, not just on the Linux runner.
# 30 KB clears both with room to spare. (Sizing to GitHub's limit alone produced
# 40 KB chunks that decoded fine on Linux and silently truncated on Windows.)
MAX_CHUNK_BYTES = 30_000
MIN_CHUNKS = 3
MAX_CHUNKS = 32  # sanity bound; beyond this, secrets are the wrong mechanism
ENV_PREFIX = "RUST_SRC_B64_"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

# Deterministic archives: identical sources must produce an identical base64
# string, so an unchanged core does not churn the secrets on every pack.
FIXED_MTIME = 0


def _collect_sources() -> list[pathlib.Path]:
    """Every Rust source under `src/`, relative to the repo root, sorted."""
    if not SRC_DIR.is_dir():
        raise SystemExit(f"no src/ directory at {SRC_DIR}")
    files = sorted(p for p in SRC_DIR.rglob("*.rs"))
    extra = sorted(p for p in SRC_DIR.rglob("*.cu"))  # CUDA kernels ship too
    out = files + extra
    if not out:
        raise SystemExit(f"no .rs/.cu files found under {SRC_DIR}")
    return out


def _build_archive() -> bytes:
    """tar+gzip the core deterministically and return the raw bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for path in _collect_sources():
            rel = path.relative_to(REPO_ROOT).as_posix()
            data = path.read_bytes()
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            info.mtime = FIXED_MTIME
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tf.addfile(info, io.BytesIO(data))
    # LZMA rather than gzip: Rust source compresses ~35% smaller, which is the
    # difference between fitting in a handful of secrets and not.
    filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
    return lzma.compress(buf.getvalue(), format=lzma.FORMAT_XZ, filters=filters)


def _decompress(archive: bytes) -> bytes:
    """Decompress an archive, accepting both the LZMA and legacy gzip formats."""
    if archive[:6] == b"\xfd7zXZ\x00":
        return lzma.decompress(archive)
    if archive[:2] == b"\x1f\x8b":
        return gzip.decompress(archive)
    raise SystemExit("unrecognised archive format (expected xz or gzip)")


def _chunk_count(b64_len: int) -> int:
    """Smallest chunk count that keeps every secret under the size limit."""
    needed = (b64_len + MAX_CHUNK_BYTES - 1) // MAX_CHUNK_BYTES
    n = max(MIN_CHUNKS, needed)
    if n > MAX_CHUNKS:
        raise SystemExit(
            f"core needs {n} secrets (> {MAX_CHUNKS}). Secrets are the wrong mechanism at "
            "this size — publish the core as an encrypted release asset instead."
        )
    return n


def _chunks(b64: str, n: int) -> list[str]:
    size = (len(b64) + n - 1) // n
    return [b64[i * size : (i + 1) * size] for i in range(n)]


def _read_chunks_from_dir(path: pathlib.Path) -> str:
    """Read `RUST_SRC_B64_1..N.txt` in order, stopping at the first gap."""
    parts = []
    for i in range(1, MAX_CHUNKS + 1):
        f = path / f"{ENV_PREFIX}{i}.txt"
        if not f.is_file():
            break
        parts.append(f.read_text(encoding="ascii").strip())
    if len(parts) < MIN_CHUNKS:
        raise SystemExit(f"found only {len(parts)} chunk file(s) in {path} (expected >= {MIN_CHUNKS})")
    return "".join(parts)


def _read_chunks_from_env() -> str:
    """Read `RUST_SRC_B64_1..N` in order, stopping at the first unset variable."""
    parts = []
    for i in range(1, MAX_CHUNKS + 1):
        val = os.environ.get(f"{ENV_PREFIX}{i}")
        if not val:
            break
        parts.append(val.strip())
    if len(parts) < MIN_CHUNKS:
        raise SystemExit(
            f"found only {len(parts)} {ENV_PREFIX}* variable(s) (expected >= {MIN_CHUNKS}). "
            "Are all the chunk secrets exposed to this job?"
        )
    return "".join(parts)


def _extract(archive: bytes, dest: pathlib.Path) -> list[str]:
    raw = _decompress(archive)
    written = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            # Path traversal guard: refuse anything escaping the destination.
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise SystemExit(f"refusing unsafe archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            target.write_bytes(extracted.read())
            written.append(member.name)
    return written


def cmd_pack(args) -> int:
    archive = _build_archive()
    b64 = base64.b64encode(archive).decode("ascii")
    digest = hashlib.sha256(archive).hexdigest()
    n = _chunk_count(len(b64))

    # Round-trip before writing anything: a corrupt pack discovered in CI costs
    # a full release cycle.
    if base64.b64decode("".join(_chunks(b64, n))) != archive:
        raise SystemExit("round-trip verification FAILED — refusing to write chunks")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Clear stale chunks from a previous, larger pack so a shrinking core cannot
    # leave an orphan file that _read_chunks_from_dir would happily concatenate.
    for stale in out.glob(f"{ENV_PREFIX}*.txt"):
        stale.unlink()

    parts = _chunks(b64, n)
    for i, part in enumerate(parts, start=1):
        f = out / f"{ENV_PREFIX}{i}.txt"
        f.write_text(part, encoding="ascii")
        print(f"  {f.name}: {len(part) / 1024:7.1f} KB")

    n_files = len(_collect_sources())
    print(f"\npacked {n_files} source files")
    print(f"archive sha256: {digest}")
    print(f"total base64:   {len(b64) / 1024:.1f} KB across {n} chunks")
    print(f"\nWrote chunks to {out}/")
    print(f"\nSet {n} GitHub Actions secrets, then delete the directory:")
    for i in range(1, n + 1):
        print(f"    gh secret set {ENV_PREFIX}{i} < \"{out}/{ENV_PREFIX}{i}.txt\"")
    print(f"\n    rm -rf \"{out}\"")
    if n > MIN_CHUNKS:
        print(
            f"\nNOTE: this pack needs {n} secrets, more than the historical 3. "
            f"Add {ENV_PREFIX}{MIN_CHUNKS + 1}..{ENV_PREFIX}{n} in repository settings; "
            "the CI restore step reads chunks until the first gap, so it adapts automatically."
        )
    return 0


def cmd_verify(args) -> int:
    b64 = _read_chunks_from_env() if args.from_env else _read_chunks_from_dir(pathlib.Path(args.inp))
    try:
        archive = base64.b64decode(b64)
    except Exception as exc:
        raise SystemExit(f"chunks do not base64-decode: {exc}") from exc

    current = _build_archive()
    packed_sha = hashlib.sha256(archive).hexdigest()
    current_sha = hashlib.sha256(current).hexdigest()
    print(f"packed  sha256: {packed_sha}")
    print(f"working sha256: {current_sha}")

    raw = _decompress(archive)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tf:
        members = [m.name for m in tf.getmembers() if m.isfile()]
    print(f"archive contains {len(members)} files")

    if packed_sha != current_sha:
        print("\nMISMATCH: the packed core differs from the working tree.")
        print("Local src/ changes will NOT reach CI until you re-pack and refresh the secrets.")
        return 1
    print("\nOK: packed core is byte-identical to the working tree.")
    return 0


def cmd_unpack(args) -> int:
    b64 = _read_chunks_from_env() if args.from_env else _read_chunks_from_dir(pathlib.Path(args.inp))
    archive = base64.b64decode(b64)
    dest = pathlib.Path(args.dest).resolve()
    written = _extract(archive, dest)
    print(f"restored {len(written)} files into {dest}")
    missing = [n for n in ("src/lib.rs",) if not (dest / n).is_file()]
    if missing:
        raise SystemExit(f"restore incomplete, missing: {missing}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack", help="pack src/ into RUST_SRC_B64_* chunk files")
    p.add_argument("--out", default=".secrets", help="output directory (default: .secrets)")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("verify", help="check packed chunks match the working tree")
    p.add_argument("--in", dest="inp", default=".secrets", help="directory holding the chunk files")
    p.add_argument("--from-env", action="store_true", help="read chunks from RUST_SRC_B64_* env vars")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("unpack", help="restore src/ from chunks (what CI runs)")
    p.add_argument("--in", dest="inp", default=".secrets", help="directory holding the chunk files")
    p.add_argument("--from-env", action="store_true", help="read chunks from RUST_SRC_B64_* env vars")
    p.add_argument("--dest", default=str(REPO_ROOT), help="checkout root to restore into")
    p.set_defaults(func=cmd_unpack)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
