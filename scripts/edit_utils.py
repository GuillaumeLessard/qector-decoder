# This script edits src/utils.rs to add estimate_distance function

with open('src/utils.rs', 'r') as f:
    content = f.read()

# Find the marker comment
marker = '/// Computes XOR detector differences across consecutive syndrome rounds:'
idx = content.find(marker)
assert idx > 0, 'Marker not found'

# Find preceding closing brace
pre = content.rfind('}', 0, idx)
assert pre > 0, 'Preceding brace not found'

insertion = '''
/// Estimate code distance from `check_to_qubits` structure.
///
/// For surface code layouts, `n_qubits = d**2`, so distance = sqrt(n_qubits).
/// This provides a more accurate estimate than volume-based heuristics.
pub fn estimate_distance(check_to_qubits: &[Vec<u32>], n_qubits: Option<usize>) -> usize {
    let nq = match n_qubits {
        Some(v) => v,
        None => {
            let mut max_q = 0u32;
            for qs in check_to_qubits {
                for &q in qs {
                    if q > max_q {
                        max_q = q;
                    }
                }
            }
            (max_q as usize) + 1
        }
    };
    if nq == 0 {
        return 0;
    }
    // For surface codes: distance = sqrt(n_qubits).
    let d = (nq as f64).sqrt().ceil() as usize;
    d.max(1)
}

/// Python wrapper for estimate_distance.
#[pyfunction]
#[pyo3(signature = (check_to_qubits, n_qubits=None))]
pub fn py_estimate_distance(check_to_qubits: Vec<Vec<u32>>, n_qubits: Option<usize>) -> usize {
    estimate_distance(&check_to_qubits, n_qubits)
}

'''

content = content[:pre+1] + insertion + content[idx:]
with open('src/utils.rs', 'w') as f:
    f.write(content)
print('SUCCESS')

