// QECTOR Decoder v3 - Rust core with PyO3 bindings
//
// This lib.rs wires all decoder modules and exposes them through a single
// PyO3 extension module for the Python package.

pub mod batch;
pub mod benchmark;
pub mod bitpack;
pub mod blossom;
pub mod bp_osd;
pub mod cascade_decoder;
pub mod cpu_batch;
pub mod cross_decoder_tests;
pub mod decoder;
pub mod fast_uf;
pub mod gf2;
pub mod gnn_graph;
pub mod gnn_layers;
pub mod gnn_predecoder;
pub mod gnn_trainer;
pub mod hybrid_decoder;
pub mod ler_benchmark;
pub mod lookup_table;
pub mod metrics;
pub mod mwpm;
pub mod neural_predecoder;
pub mod safetensors_loader;
pub mod sliding_window;
pub mod sparse_blossom;
pub mod streaming;
pub mod uf_core;
pub mod utils;

#[cfg(feature = "cuda")]
pub mod cuda_batch;
#[cfg(all(test, feature = "cuda"))]
pub mod cuda_batch_tests;
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
pub use sliding_window::SlidingWindowDecoder;
pub use sparse_blossom::SparseBlossomDecoder;
pub use streaming::StreamingDecoder;

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
    m.add_class::<lookup_table::PyLookupTableDecoder>()?;
    m.add_class::<neural_predecoder::PyNeuralPredecoder>()?;

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
    m.add_function(wrap_pyfunction!(utils::py_generate_repetition_code_checks, m)?)?;

    // Metrics
    m.add_function(wrap_pyfunction!(metrics::start_metrics_server, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::get_latency_quantiles, m)?)?;

    // MCP server
    m.add_function(wrap_pyfunction!(mcp_server::run_mcp_server, m)?)?;

    // Optional: CUDA
    #[cfg(feature = "cuda")]
    {
        m.add_class::<cuda_python::PyCUDABatchDecoder>()?;
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
