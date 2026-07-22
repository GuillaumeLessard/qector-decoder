use pyo3::prelude::*;

pub mod batch;
pub mod benchmark;
pub mod blossom;
pub mod bp_osd;
pub mod cascade_decoder;
pub mod cpu_batch;
#[cfg(feature = "cuda")]
pub mod cuda_batch;
#[cfg(feature = "cuda")]
mod cuda_graph;
#[cfg(feature = "cuda")]
mod cuda_python;
#[cfg(feature = "cuda")]
mod cuda_runtime;
#[cfg(feature = "cuda")]
mod cuda_workspace;
pub mod decoder;
pub mod fast_uf;
pub mod gnn_graph;
pub mod gnn_layers;
pub mod gnn_predecoder;
pub mod gnn_trainer;
pub mod hybrid_decoder;
#[cfg(feature = "grpc")]
pub mod grpc_server;
pub mod ler_benchmark;
pub mod lookup_table;
pub mod mcp_server;
#[cfg(feature = "grpc")]
pub mod metrics;
pub mod mwpm;
pub mod neural_predecoder;
pub mod opencl_batch;
mod safetensors_loader;
pub mod sliding_window;
pub mod sparse_blossom;
pub mod streaming;
pub mod uf_core;
pub mod utils;

use pyo3::wrap_pyfunction;

#[cfg(test)]
#[path = "cross_decoder_tests.rs"]
mod cross_decoder_tests;

/// QECTOR Decoder v3 — Source-available Rust QEC decoder core.
/// Rust core with PyO3 Python bindings.
#[pymodule]
fn qector_decoder_v3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Decoder classes
    m.add_class::<blossom::PyBlossomDecoder>()?;
    m.add_class::<fast_uf::PyFastUnionFindDecoder>()?;
    m.add_class::<decoder::PyUnionFindDecoder>()?;
    m.add_class::<sparse_blossom::PySparseBlossomDecoder>()?;
    m.add_class::<sliding_window::PySlidingWindowDecoder>()?;
    m.add_class::<streaming::PyStreamingDecoder>()?;
    m.add_class::<batch::PyBatchDecoder>()?;
    m.add_class::<cpu_batch::PyCPUBatchDecoder>()?;
    #[cfg(feature = "cuda")]
    m.add_class::<cuda_python::PyCUDABatchDecoder>()?;
    m.add_class::<benchmark::PyBenchmarkSuite>()?;
    m.add_class::<ler_benchmark::PyLERBenchmark>()?;
    m.add_class::<lookup_table::PyLookupTableDecoder>()?;
    m.add_class::<bp_osd::PyBPOSDDecoder>()?;
    m.add_class::<neural_predecoder::PyNeuralPredecoder>()?;
    m.add_class::<gnn_graph::PyDetectorGraph>()?;
    m.add_class::<gnn_predecoder::PyGNNPredecoder>()?;
    m.add_class::<gnn_trainer::PyGNNTrainer>()?;
    #[cfg(feature = "opencl")]
    m.add_class::<opencl_batch::PyOpenCLBatchDecoder>()?;
    m.add_class::<hybrid_decoder::PyHybridDecoder>()?;
    m.add_class::<cascade_decoder::PyHybridCascadeDecoder>()?;

    // Utility functions
    m.add_wrapped(wrap_pyfunction!(utils::py_check_to_edges))?;
    m.add_wrapped(wrap_pyfunction!(utils::py_generate_surface_code_checks))?;
    m.add_wrapped(wrap_pyfunction!(utils::py_generate_toy_code_checks))?;
    m.add_wrapped(wrap_pyfunction!(utils::py_generate_ring_code_checks))?;
    m.add_wrapped(wrap_pyfunction!(utils::py_generate_repetition_code_checks))?;

    // Infrastructure servers (gRPC + Prometheus)
    #[cfg(feature = "grpc")]
    m.add_wrapped(wrap_pyfunction!(metrics::start_metrics_server))?;
    #[cfg(feature = "grpc")]
    m.add_wrapped(wrap_pyfunction!(grpc_server::run_grpc_server))?;
    m.add_wrapped(wrap_pyfunction!(mcp_server::run_mcp_server))?;

    #[cfg(feature = "opencl")]
    m.add_wrapped(wrap_pyfunction!(opencl_batch::py_opencl_is_available))?;
    #[cfg(feature = "cuda")]
    m.add_wrapped(wrap_pyfunction!(cuda_python::py_cuda_is_available))?;

    Ok(())
}
