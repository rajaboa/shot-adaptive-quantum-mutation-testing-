"""QAOA circuit for MaxCut on small graphs.

Produces non-trivial probability distributions over multiple outcomes,
making it a good target for adaptive shot mutation testing.
"""

from qiskit import QuantumCircuit


def build_circuit(gamma=0.8, beta=0.4, n_qubits=4, p=1, measurement=True):
    """Build a p-layer QAOA circuit for MaxCut on a ring graph.

    Args:
        gamma: Problem unitary angle (one per layer, or single value for all).
        beta: Mixer unitary angle (one per layer, or single value for all).
        n_qubits: Number of qubits (nodes in the ring graph).
        p: Number of QAOA layers.
        measurement: Whether to include measurements.

    Returns:
        QuantumCircuit with a non-trivial output distribution.
    """
    # Support both scalar and list parameters
    gammas = [gamma] * p if not isinstance(gamma, list) else gamma
    betas = [beta] * p if not isinstance(beta, list) else beta

    # Ring graph edges: (0,1), (1,2), ..., (n-2,n-1), (n-1,0)
    edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]

    qc = QuantumCircuit(n_qubits, n_qubits if measurement else 0)

    # Initial superposition
    for i in range(n_qubits):
        qc.h(i)

    for layer in range(p):
        g = gammas[layer]
        b = betas[layer]

        # Problem unitary: exp(-i * gamma * C)
        # For each edge (i,j): exp(-i * gamma * Z_i Z_j)
        #   = CNOT(i,j) . Rz(2*gamma, j) . CNOT(i,j)
        for i, j in edges:
            qc.cx(i, j)
            qc.rz(2 * g, j)
            qc.cx(i, j)

        # Mixer unitary: exp(-i * beta * B)
        # B = sum_i X_i, so exp(-i*beta*X_i) = Rx(2*beta, i)
        for i in range(n_qubits):
            qc.rx(2 * b, i)

    if measurement:
        qc.measure(range(n_qubits), range(n_qubits))

    return qc
