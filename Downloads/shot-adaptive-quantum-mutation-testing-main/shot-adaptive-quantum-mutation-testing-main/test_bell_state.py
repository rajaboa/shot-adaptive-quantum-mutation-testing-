import unittest
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from bell_state import bell_state


class TestBellState(unittest.TestCase):

    def test_bell_circuit_has_2_qubits(self):
        qc = bell_state()
        self.assertEqual(qc.num_qubits, 2)

    def test_bell_circuit_has_h_gate(self):
        qc = bell_state()
        gate_names = [instr.operation.name for instr in qc.data]
        self.assertIn('h', gate_names)

    def test_bell_circuit_has_cx_gate(self):
        qc = bell_state()
        gate_names = [instr.operation.name for instr in qc.data]
        self.assertIn('cx', gate_names)

    def test_bell_circuit_produces_entangled_state(self):
        qc = bell_state()
        simulator = AerSimulator()
        job = simulator.run(qc, shots=1000)
        counts = job.result().get_counts()
        # Bell state should only produce |00> and |11>
        for bitstring in counts:
            self.assertIn(bitstring, ['00', '11'])


if __name__ == '__main__':
    unittest.main()
