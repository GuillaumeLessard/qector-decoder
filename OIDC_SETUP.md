# PyPI OIDC Trusted Publisher Setup

Complete these steps **once** in a browser before pushing the first release tag.

---

## Step 1: PyPI Trusted Publisher (2 minutes)

1. Go to <https://pypi.org/manage/account/publishing/>
2. Log in with the `qectorlab` / `admin@qector.store` account
3. Under **"Add a new pending publisher"**, fill in:

   | Field | Value |
   |-------|-------|
   | PyPI Project Name | `qector-decoder-v3` |
   | Owner | `qectorlab` |
   | Repository name | `qector-decoder` |
   | Workflow filename | `CI.yml` |
   | Environment name | `pypi` |

4. Click **Add**.

---

## Step 2: GitHub Environment (1 minute)

1. Go to <https://github.com/qectorlab/qector-decoder/settings/environments>
2. Click **New environment**
3. Name it exactly: `pypi`
4. Click **Configure environment**
5. Under **Deployment branches and tags**, select **Selected branches and tags**
6. Add rule: `refs/tags/v*`
7. Save.

---

## Step 3: Add GitHub Secrets (2 minutes)

The Rust source archive is split across two secrets.

1. Go to <https://github.com/qectorlab/qector-decoder/settings/secrets/actions>
2. Add secret **`RUST_SRC_B64_1`**:
   - Content: copy the **entire content** of `C:\Users\Admin\Desktop\qector_src_b64_part1.txt`
3. Add secret **`RUST_SRC_B64_2`**:
   - Content: copy the **entire content** of `C:\Users\Admin\Desktop\qector_src_b64_part2.txt`

> The CI workflow concatenates `RUST_SRC_B64_1 + RUST_SRC_B64_2`, base64-decodes
> the result, and extracts `src/`, `build.rs`, and `proto/` before running maturin.
> Forks and external PRs will not have these secrets so their wheel builds are
> gracefully skipped (lint and Docker checks still run).

---

## Step 4: Regenerate the secret archive (when source changes)

```powershell
# Run from C:\Users\Admin\Desktop\qector-decoder-main
tar -czf C:\Users\Admin\Desktop\qector_src_only.tar.gz `
    -C C:\Users\Admin\Desktop\qector-decoder-main src build.rs proto

$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\Users\Admin\Desktop\qector_src_only.tar.gz"))
$half = [int]($b64.Length / 2)
[IO.File]::WriteAllText("C:\Users\Admin\Desktop\qector_src_b64_part1.txt", $b64.Substring(0, $half))
[IO.File]::WriteAllText("C:\Users\Admin\Desktop\qector_src_b64_part2.txt", $b64.Substring($half))
```

Then update both GitHub secrets with the new content.

---

## Step 5: Trigger a release

```powershell
cd C:\Users\Admin\Desktop\qector-push-clean
git tag v0.5.0
git push origin v0.5.0
```

CI will build wheels for all platforms and publish to PyPI automatically.
