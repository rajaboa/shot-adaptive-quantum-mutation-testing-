""" Test IQPE """

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'programs'))

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit_algorithms import NumPyMinimumEigensolver
from iqpe import IQPE
from quantum_instance import QuantumInstance


class TestIQPE(unittest.TestCase):
    """IQPE tests."""

    def setUp(self):
        # Simple 1-qubit Hamiltonian: X + Y + Z + I
        self.qubit_op_simple = SparsePauliOp.from_list([
            ('X', 1.0), ('Y', 1.0), ('Z', 1.0), ('I', 1.0)
        ])

        # ZZ operator
        self.qubit_op_zz = SparsePauliOp.from_list([('ZZ', 1.0)])

        # H2 with 2 qubit reduction
        self.qubit_op_h2 = SparsePauliOp.from_list([
            ('II', -1.052373245772859),
            ('IZ', 0.39793742484318045),
            ('ZI', -0.39793742484318045),
            ('ZZ', -0.01128010425623538),
            ('XX', 0.18093119978423156),
        ])

    def test_iqpe_simple_qasm(self):
        """IQPE test with simple operator on qasm simulator"""
        qubit_op = self.qubit_op_simple
        # Get exact eigenvalue
        exact = NumPyMinimumEigensolver()
        exact_result = exact.compute_minimum_eigenvalue(qubit_op)
        ref_eigenval = exact_result.eigenvalue

        # Prepare eigenstate
        ref_eigenvec = exact_result.eigenstate
        state_in = QuantumCircuit(qubit_op.num_qubits)
        state_in.initialize(ref_eigenvec, state_in.qubits)

        iqpe = IQPE(qubit_op, state_in, num_time_slices=1, num_iterations=5,
                    expansion_mode='suzuki', expansion_order=2,
                    shallow_circuit_concat=True)

        backend = AerSimulator()
        qi = QuantumInstance(backend, shots=100)
        result = iqpe.run(qi)

        np.testing.assert_approx_equal(result.eigenvalue.real, ref_eigenval.real, significant=2)

    def test_iqpe_zz_sv(self):
        """IQPE test with ZZ operator on statevector simulator"""
        qubit_op = self.qubit_op_zz
        exact = NumPyMinimumEigensolver()
        exact_result = exact.compute_minimum_eigenvalue(qubit_op)
        ref_eigenval = exact_result.eigenvalue

        ref_eigenvec = exact_result.eigenstate
        state_in = QuantumCircuit(qubit_op.num_qubits)
        state_in.initialize(ref_eigenvec, state_in.qubits)

        iqpe = IQPE(qubit_op, state_in, num_time_slices=1, num_iterations=1,
                    expansion_mode='suzuki', expansion_order=2,
                    shallow_circuit_concat=True)

        backend = StatevectorSimulator()
        qi = QuantumInstance(backend)
        result = iqpe.run(qi)

        np.testing.assert_approx_equal(result.eigenvalue.real, ref_eigenval.real, significant=2)

    def test_iqpe_h2_sv(self):
        """IQPE test with H2 operator on statevector simulator"""
        qubit_op = self.qubit_op_h2
        exact = NumPyMinimumEigensolver()
        exact_result = exact.compute_minimum_eigenvalue(qubit_op)
        ref_eigenval = exact_result.eigenvalue

        ref_eigenvec = exact_result.eigenstate
        state_in = QuantumCircuit(qubit_op.num_qubits)
        state_in.initialize(ref_eigenvec, state_in.qubits)

        iqpe = IQPE(qubit_op, state_in, num_time_slices=1, num_iterations=6,
                    expansion_mode='suzuki', expansion_order=2,
                    shallow_circuit_concat=True)

        backend = StatevectorSimulator()
        qi = QuantumInstance(backend)
        result = iqpe.run(qi)

        np.testing.assert_approx_equal(result.eigenvalue.real, ref_eigenval.real, significant=2)


if __name__ == '__main__':
    unittest.main()
