"""GHZ program with inline imperative gate construction for QMutPy AST mutation.

The body of build_circuit() mirrors the upstream mqt.bench `ghz.create_circuit`
so that QMutPy's quantum operators (QGD/QGR/QGI) have visible `.h()` / `.cx()`
calls to mutate. The final transpile() reproduces mqt.bench's
BenchmarkLevel.INDEP step (optimization_level=2, seed_transpiler=10) so that
the resulting circuit matches the one baselines in
test_cases_official/ghz_test_cases.json were generated against.
"""
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister

ALGORITHM_NAME = "ghz"
AVAILABLE_QUBITS = list(range(2, 11))


def build_circuit(n_qubits, measurement=True):
    if n_qubits < 2:
        raise ValueError(
            f"{ALGORITHM_NAME}: n_qubits={n_qubits} too small. Minimum is 2."
        )
    if n_qubits >= 11:
        raise ValueError(
            f"{ALGORITHM_NAME}: n_qubits={n_qubits} too large. Maximum is 10."
        )
    if n_qubits not in AVAILABLE_QUBITS:
        raise ValueError(
            f"{ALGORITHM_NAME}: n_qubits={n_qubits} not available. "
            f"Available sizes: {AVAILABLE_QUBITS}"
        )

    q = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(q, name="ghz")
    qc.h(q[-1])
    for i in range(1, n_qubits):
        qc.cx(q[n_qubits - i], q[n_qubits - i - 1])
    if measurement:
        qc.measure_all()

    return transpile(qc, optimization_level=2, seed_transpiler=10)
