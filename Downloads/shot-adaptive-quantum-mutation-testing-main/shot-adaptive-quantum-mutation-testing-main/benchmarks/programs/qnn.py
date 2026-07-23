"""Program for qnn using official mqt.bench package.

This module has inline circuit construction logic for AST-level mutation,
and produces circuits for circuit-level mutation.
"""
from mqt.bench import get_benchmark, BenchmarkLevel

ALGORITHM_NAME = "qnn"
AVAILABLE_QUBITS = list(range(2, 5))

def build_circuit(n_qubits, measurement=True):
    """Build qnn circuit using mqt.bench package.
    
    Args:
        n_qubits: Number of qubits for the circuit
        measurement: Whether to include measurement (ignored, mqt.bench includes it)
        
    Returns:
        QuantumCircuit: The generated circuit
        
    Raises:
        ValueError: If n_qubits is not in available range
    """
    # Validate qubit count (this logic can be mutated)
    if n_qubits < 2:
        raise ValueError(
            f"{ALGORITHM_NAME}: n_qubits={n_qubits} too small. "
            f"Minimum is 2."
        )
    
    if n_qubits >= 5:
        raise ValueError(
            f"{ALGORITHM_NAME}: n_qubits={n_qubits} too large. "
            f"Maximum is 4."
        )
    
    # Check if qubit count is available (this logic can be mutated)
    if n_qubits not in AVAILABLE_QUBITS:
        raise ValueError(
            f"{ALGORITHM_NAME}: n_qubits={n_qubits} not available. "
            f"Available sizes: {AVAILABLE_QUBITS}"
        )
    
    # Generate the circuit using mqt.bench
    circuit = get_benchmark(
        benchmark="qnn",
        level=BenchmarkLevel.INDEP,
        circuit_size=n_qubits
    )
    
    return circuit
