"""qpeinexact program with inline imperative gate construction for QMutPy AST mutation.

Body mirrors upstream mqt.bench `qpeinexact.create_circuit`. The final
transpile() reproduces BenchmarkLevel.INDEP (optimization_level=2,
seed_transpiler=10) so the resulting circuit matches the baselines in
test_cases_official/qpeinexact_test_cases.json.

`synth_qft_full(...)` is composed at the end and is a synthesis primitive that
QMutPy cannot see inside. The x/h/cp prelude is the mutate-able part.
"""
import random
from fractions import Fraction

import numpy as np
from qiskit import transpile
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.synthesis import synth_qft_full

ALGORITHM_NAME = "qpeinexact"
AVAILABLE_QUBITS = list(range(5, 14))


def build_circuit(n_qubits, measurement=True):
    if n_qubits < 5:
        raise ValueError(
            f"{ALGORITHM_NAME}: n_qubits={n_qubits} too small. Minimum is 5."
        )
    if n_qubits >= 14:
        raise ValueError(
            f"{ALGORITHM_NAME}: n_qubits={n_qubits} too large. Maximum is 13."
        )
    if n_qubits not in AVAILABLE_QUBITS:
        raise ValueError(
            f"{ALGORITHM_NAME}: n_qubits={n_qubits} not available. "
            f"Available sizes: {AVAILABLE_QUBITS}"
        )

    num_qubits = n_qubits - 1  # because of ancilla qubit
    q = QuantumRegister(num_qubits, "q")
    psi = QuantumRegister(1, "psi")
    c = ClassicalRegister(num_qubits, "c")
    qc = QuantumCircuit(q, psi, c, name="qpeinexact")

    random.seed(10)
    theta = 0
    while theta == 0 or (theta & 1) == 0:
        theta = random.getrandbits(num_qubits + 1)
    lam = Fraction(0, 1)
    for i in range(num_qubits + 1):
        if theta & (1 << (num_qubits - i)):
            lam += Fraction(1, (1 << i))

    qc.x(psi)
    qc.h(q)

    for i in range(num_qubits):
        angle = (lam * (1 << i)) % 2
        if angle > 1:
            angle -= 2
        if angle != 0:
            qc.cp(angle * np.pi, psi, q[i])

    qc.compose(
        synth_qft_full(num_qubits=num_qubits, inverse=True),
        inplace=True,
        qubits=list(range(num_qubits)),
    )
    qc.barrier()
    if measurement:
        qc.measure(q, c)

    return transpile(qc, optimization_level=2, seed_transpiler=10)
