"""qwalk program with inline imperative gate construction for QMutPy AST mutation.

Body mirrors upstream mqt.bench `qwalk.create_circuit`. The final transpile()
reproduces BenchmarkLevel.INDEP (optimization_level=2, seed_transpiler=10) so
the resulting circuit matches the baselines in
test_cases_official/qwalk_test_cases.json.

Note: `mcx` is not in QMutPy's hardcoded gate list, so QGD/QGR/QGI will skip
those calls. The h/cx/x calls are mutate-able.
"""
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister

ALGORITHM_NAME = "qwalk"
AVAILABLE_QUBITS = list(range(3, 6))


def build_circuit(n_qubits, measurement=True):
    if n_qubits < 3:
        raise ValueError(
            f"{ALGORITHM_NAME}: n_qubits={n_qubits} too small. Minimum is 3."
        )
    if n_qubits >= 6:
        raise ValueError(
            f"{ALGORITHM_NAME}: n_qubits={n_qubits} too large. Maximum is 5."
        )
    if n_qubits not in AVAILABLE_QUBITS:
        raise ValueError(
            f"{ALGORITHM_NAME}: n_qubits={n_qubits} not available. "
            f"Available sizes: {AVAILABLE_QUBITS}"
        )

    num_qubits = n_qubits - 1  # because one qubit is needed for the coin
    depth = 3
    coin = QuantumRegister(1, "coin")
    node = QuantumRegister(num_qubits, "node")

    qc = QuantumCircuit(node, coin, name="qwalk")

    for _ in range(depth):
        qc.h(coin)

        for i in range(num_qubits - 1):
            qc.mcx(coin[:] + node[i + 1 :], node[i])
        qc.cx(coin, node[num_qubits - 1])

        qc.x(coin)
        qc.x(node[1:])
        for i in range(num_qubits - 1):
            qc.mcx(coin[:] + node[i + 1 :], node[i])
        qc.cx(coin, node[num_qubits - 1])
        qc.x(node[1:])
        qc.x(coin)

    if measurement:
        qc.measure_all()

    return transpile(qc, optimization_level=2, seed_transpiler=10)
