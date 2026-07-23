"""wstate program with inline imperative gate construction for QMutPy AST mutation.

Body mirrors upstream mqt.bench `wstate.create_circuit`. The final transpile()
reproduces BenchmarkLevel.INDEP (optimization_level=2, seed_transpiler=10) so
the resulting circuit matches the baselines in
test_cases_official/wstate_test_cases.json.

All gates (x/ry/cz/cx) are mutate-able by QMutPy, including those inside the
nested f_gate helper.
"""
import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister

ALGORITHM_NAME = "wstate"
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

    num_qubits = n_qubits
    q = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(q, name="wstate")

    def f_gate(qc, q, i, j, n, k):
        theta = np.arccos(np.sqrt(1 / (n - k + 1)))
        qc.ry(-theta, q[j])
        qc.cz(q[i], q[j])
        qc.ry(theta, q[j])

    qc.x(q[-1])

    for m in range(1, num_qubits):
        f_gate(qc, q, num_qubits - m, num_qubits - m - 1, num_qubits, m)

    for k in reversed(range(1, num_qubits)):
        qc.cx(k - 1, k)

    if measurement:
        qc.measure_all()

    return transpile(qc, optimization_level=2, seed_transpiler=10)
