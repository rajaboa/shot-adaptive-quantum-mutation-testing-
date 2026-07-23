"""Minimal QuantumInstance replacement for Qiskit 2.x compatibility."""

from qiskit import transpile


class QuantumInstance:
    """Simple wrapper around a backend, replacing the old qiskit.aqua.QuantumInstance."""

    def __init__(self, backend, shots=1024, seed_simulator=None, seed_transpiler=None):
        self.backend = backend
        self.shots = shots
        self.seed_simulator = seed_simulator
        self.seed_transpiler = seed_transpiler
        backend_name = getattr(backend, 'name', '')
        if callable(backend_name):
            backend_name = backend_name()
        self.is_statevector = 'statevector' in backend_name

    def execute(self, circuit):
        qc = transpile(circuit, self.backend, seed_transpiler=self.seed_transpiler)
        if self.is_statevector:
            return self.backend.run(qc, seed_simulator=self.seed_simulator).result()
        else:
            return self.backend.run(
                qc, shots=self.shots, seed_simulator=self.seed_simulator
            ).result()
