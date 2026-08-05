// QECTOR Decoder v3 - Rust core with PyO3 bindings
//
// This lib.rs wires all decoder modules and exposes them through a single
// PyO3 extension module for the Python package.

// Allow specific clippy warnings in proprietary core (cannot edit .rs files directly)
#![allow(unused_doc_comments, unused_imports, renamed_and_removed_lints)]

pub mod ambig_cluster;
pub mod auto_decoder;
pub mod batch;
pub mod benchmark;
pub mod bitpack;
pub mod blossom;
pub mod bp_osd;
pub mod cascade_decoder;
pub mod core;
pub mod cpu_batch;
pub mod cross_decoder_tests;
pub mod decoder;
/// Test-only loader for circuit-level DEM benchmark fixtures (see UF-02).
#[cfg(test)]
pub mod dem_fixture;
pub mod fast_uf;
pub mod fusion_mwpm;
pub mod gf2;
pub mod gnn_graph;
pub mod gnn_layers;
pub mod gnn_predecoder;
pub mod gnn_trainer;
pub mod hybrid_decoder;
pub mod ler_benchmark;
pub mod license;
pub mod lookup_table;
pub mod metrics;
pub mod mwpm;
pub mod neural_predecoder;
pub mod safetensors_loader;
pub mod sliding_window;
pub mod space_time_decoder;
pub mod sparse_blossom;
pub mod streaming;
pub mod stripe_billing;
pub mod two_stage_decoder;
pub mod uf_core;
pub mod utils;

#[cfg(feature = "cuda")]
pub mod cuda_batch;
// NOTE: `cuda_batch_tests.rs` is included exactly once, by `cuda_batch.rs`
// via `#[path] mod tests` — declaring it here too makes clippy's
// `duplicate_mod` fire when the `cuda` feature is enabled.
#[cfg(feature = "cuda")]
pub mod cuda_bp_osd;
#[cfg(feature = "cuda")]
pub mod cuda_graph;
#[cfg(feature = "cuda")]
pub mod cuda_python;
#[cfg(feature = "cuda")]
pub mod cuda_runtime;
#[cfg(feature = "cuda")]
pub mod cuda_workspace;

#[cfg(feature = "opencl")]
pub mod opencl_batch;

#[cfg(feature = "grpc")]
pub mod grpc_server;

pub mod mcp_server;

// Re-export core types for downstream Rust users
pub use ambig_cluster::AmbiguityClusterDecoder;
pub use auto_decoder::AutoDecoder;
pub use batch::BatchDecoder;
pub use benchmark::BenchmarkSuite;
pub use blossom::BlossomDecoder;
pub use bp_osd::BPOSDDecoder;
pub use cascade_decoder::HybridCascadeDecoder;
pub use cpu_batch::CPUBatchDecoder;
pub use decoder::UnionFindDecoder;
pub use fast_uf::FastUnionFindDecoder;
pub use gnn_graph::DetectorGraph;
pub use gnn_predecoder::GNNPredecoder;
pub use gnn_trainer::GNNTrainer;
pub use hybrid_decoder::HybridDecoder;
pub use ler_benchmark::LERBenchmark;
pub use lookup_table::LookupTableDecoder;
pub use neural_predecoder::NeuralPredecoder;
#[allow(deprecated)]
pub use sliding_window::SlidingWindowDecoder;
pub use sparse_blossom::SparseBlossomDecoder;
#[allow(deprecated)]
pub use streaming::StreamingDecoder;
pub use two_stage_decoder::TwoStageDecoder;

use pyo3::prelude::*;

#[pymodule]
fn qector_decoder_v3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Core decoders
    m.add_class::<decoder::PyUnionFindDecoder>()?;
    m.add_class::<fast_uf::PyFastUnionFindDecoder>()?;
    m.add_class::<blossom::PyBlossomDecoder>()?;
    m.add_class::<sparse_blossom::PySparseBlossomDecoder>()?;
    m.add_class::<bp_osd::PyBPOSDDecoder>()?;
    m.add_class::<cascade_decoder::PyHybridCascadeDecoder>()?;
    m.add_class::<hybrid_decoder::PyHybridDecoder>()?;
    m.add_class::<sliding_window::PySlidingWindowDecoder>()?;
    m.add_class::<streaming::PyStreamingDecoder>()?;
    m.add_class::<space_time_decoder::PySpaceTimeDecoder>()?;
    m.add_class::<lookup_table::PyLookupTableDecoder>()?;
    m.add_class::<neural_predecoder::PyNeuralPredecoder>()?;
    m.add_class::<auto_decoder::PyAutoDecoder>()?;
    m.add_class::<two_stage_decoder::PyTwoStageDecoder>()?;
    m.add_class::<ambig_cluster::PyAmbiguityClusterDecoder>()?;

    // Batch decoders
    m.add_class::<batch::PyBatchDecoder>()?;
    m.add_class::<cpu_batch::PyCPUBatchDecoder>()?;

    // GNN
    m.add_class::<gnn_graph::PyDetectorGraph>()?;
    m.add_class::<gnn_predecoder::PyGNNPredecoder>()?;
    m.add_class::<gnn_trainer::PyGNNTrainer>()?;

    // Benchmarking
    m.add_class::<benchmark::PyBenchmarkSuite>()?;
    m.add_class::<ler_benchmark::PyLERBenchmark>()?;

    // Utility functions
    m.add_function(wrap_pyfunction!(utils::py_check_to_edges, m)?)?;
    m.add_function(wrap_pyfunction!(utils::py_generate_parity_check_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(utils::py_generate_ring_code_checks, m)?)?;
    m.add_function(wrap_pyfunction!(utils::py_generate_surface_code_checks, m)?)?;
    m.add_function(wrap_pyfunction!(utils::py_generate_toy_code_checks, m)?)?;
    m.add_function(wrap_pyfunction!(
        utils::py_generate_repetition_code_checks,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(utils::py_compute_detector_differences, m)?)?;
    m.add_function(wrap_pyfunction!(
        utils::py_generate_space_time_surface_code_checks,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        utils::py_generate_triangular_color_code_4_8_8_checks,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        utils::py_generate_biconnected_qldpc_checks,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(utils::py_estimate_distance, m)?)?;

    // SIMD buffer geometry (P-04): inspect what the decoder sees for a given
    // NumPy buffer, and allocate one with a 64-byte-multiple row stride.
    m.add_function(wrap_pyfunction!(auto_decoder::syndrome_buffer_geometry, m)?)?;
    m.add_function(wrap_pyfunction!(auto_decoder::aligned_syndrome_buffer, m)?)?;

    // License & Stripe
    m.add_function(wrap_pyfunction!(license::py_set_license_key, m)?)?;
    m.add_function(wrap_pyfunction!(license::py_get_license_info, m)?)?;
    m.add_function(wrap_pyfunction!(license::py_enforce_distance_cap, m)?)?;
    m.add_function(wrap_pyfunction!(license::py_enforce_unlocked, m)?)?;
    m.add_function(wrap_pyfunction!(stripe_billing::py_record_shots, m)?)?;
    m.add_function(wrap_pyfunction!(
        stripe_billing::py_get_accumulated_shots,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stripe_billing::py_flush_usage, m)?)?;

    // Metrics
    m.add_function(wrap_pyfunction!(metrics::start_metrics_server, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::get_latency_quantiles, m)?)?;

    // MCP server
    m.add_function(wrap_pyfunction!(mcp_server::run_mcp_server, m)?)?;

    // Optional: CUDA
    #[cfg(feature = "cuda")]
    {
        m.add_class::<cuda_python::PyCUDABatchDecoder>()?;
        m.add_class::<cuda_python::PyCUDABpOsdDecoder>()?;
        m.add_function(wrap_pyfunction!(cuda_python::py_cuda_is_available, m)?)?;
    }

    // Optional: OpenCL
    #[cfg(feature = "opencl")]
    {
        m.add_class::<opencl_batch::PyOpenCLBatchDecoder>()?;
        m.add_function(wrap_pyfunction!(opencl_batch::py_opencl_is_available, m)?)?;
    }

    // Optional: gRPC
    #[cfg(feature = "grpc")]
    {
        m.add_function(wrap_pyfunction!(grpc_server::run_grpc_server, m)?)?;
        m.add_function(wrap_pyfunction!(grpc_server::start_grpc_server, m)?)?;
    }

    Ok(())
}
