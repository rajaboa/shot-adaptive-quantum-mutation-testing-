"""qftentangled program with inline imperative gate construction for QMutPy AST mutation.

Body mirrors upstream mqt.bench `qftentangled.create_circuit`. The final
transpile() reproduces BenchmarkLevel.INDEP (optimization_level=2,
seed_transpiler=10) so the resulting circuit matches the baselines in
test_cases_official/qftentangled_test_cases.json.

Note: `QFTGate(num_qubits=...)` is a Qiskit circuit-library object that QMutPy
cannot see inside. Only the h/cx prelude is mutate-able at the AST level.
"""
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFTGate

ALGORITHM_NAME = "qftentangled"
AVAILABLE_QUBITS = list(range(3, 11))


def build_circuit(n_qubits, measurement=True):
    if n_qubits < 3:
        raise ValueError(
            f"{ALGORITHM_NAME}: n_qubits={n_qubits} too small. Minimum is 3."
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

    num_qubits = n_qubits
    q = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(q)
    qc.h(q[-1])
    for i in range(1, num_qubits):
        qc.cx(q[num_qubits - i], q[num_qubits - i - 1])

    qc.compose(QFTGate(num_qubits=num_qubits), inplace=True)

    if measurement:
        qc.measure_all()
    qc.name = "qftentangled"

    return transpile(qc, optimization_level=2, seed_transpiler=10)
