//! Build script for the qector_decoder_v3 extension.
//!
//! Responsibilities:
//!   * When (and only when) the `grpc` feature is enabled (`grpc` / `full`),
//!     compile the gRPC protobuf (`proto/qector.proto`) via `tonic_prost_build`.
//!     `protoc` is located from the bundled `protoc_dir/` first, then `PATH`,
//!     then a vendored binary (`protoc-bin-vendored`) so the documented
//!     `--features full` build works with no system install. If the grpc
//!     feature is on and the proto still cannot be compiled, the build fails
//!     with an actionable error instead of a cryptic missing-file error later.
//!   * Add `lib/` to the native link search path (Windows OpenCL import lib, etc.).
//!
//! The default CPU build and the CUDA build compile no protobuf and do NOT
//! require `protoc`. GPU backends (CUDA/OpenCL) are selected by Cargo features
//! and load their drivers at runtime; nothing GPU-specific is linked here. The
//! CUDA kernel (`src/cuda_kernels.cu`) is compiled at runtime via NVRTC, so it
//! is only listed as a `rerun-if-changed` input.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda_kernels.cu");
    println!("cargo:rerun-if-changed=proto/qector.proto");

    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string()));

    // The gRPC protobuf is only needed when the `grpc` feature is active
    // (`--features grpc` / `--features full`). The default CPU build and the
    // CUDA build skip it entirely, so they never require `protoc`.
    if std::env::var_os("CARGO_FEATURE_GRPC").is_some() {
        compile_grpc_proto(&manifest_dir);
    }

    // Native link search path: bundled import libraries (e.g. Windows OpenCL).
    println!(
        "cargo:rustc-link-search=native={}",
        manifest_dir.join("lib").display()
    );
}

/// Locate a `protoc` and compile `proto/qector.proto`. Preference order:
/// bundled copy (`protoc_dir/`), then `PATH`, then the vendored binary. Panics
/// with an actionable message if the proto cannot be compiled, because the
/// `grpc` feature cannot build without the generated code.
fn compile_grpc_proto(manifest_dir: &Path) {
    let bundled = manifest_dir.join("protoc_dir/bin/protoc.exe");
    if bundled.exists() {
        println!("cargo:warning=Using bundled protoc: {}", bundled.display());
        std::env::set_var("PROTOC", &bundled);
    } else if Command::new("protoc").arg("--version").output().is_ok() {
        // protoc already on PATH — tonic_prost_build picks it up via the default lookup.
    } else {
        // Fall back to the vendored protoc so `--features full` builds with no
        // system install.
        match protoc_bin_vendored::protoc_bin_path() {
            Ok(path) => {
                println!("cargo:warning=Using vendored protoc: {}", path.display());
                std::env::set_var("PROTOC", path);
            }
            Err(e) => panic!(
                "the `grpc` feature requires `protoc` to compile proto/qector.proto, but none \
                 was found on PATH and no vendored protoc is available for this platform ({e}). \
                 Install protoc (https://protobuf.dev/downloads) and rebuild, or build without \
                 the `grpc`/`full` feature."
            ),
        }
    }

    if let Err(e) = tonic_prost_build::compile_protos("proto/qector.proto") {
        panic!(
            "failed to compile proto/qector.proto for the `grpc` feature: {e}. Ensure a working \
             `protoc` is available (https://protobuf.dev/downloads)."
        );
    }
}
