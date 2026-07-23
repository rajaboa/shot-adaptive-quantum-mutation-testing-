""" test Grover """

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'programs'))

from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator, StatevectorSimulator
from grover import Grover, GroverResult
from quantum_instance import QuantumInstance


class TestGroverConstructor(unittest.TestCase):
    """Test for the constructor of Grover"""

    def setUp(self):
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        self._expected_grover_op = GroverOperator(oracle=oracle)

    def test_oracle_quantumcircuit(self):
        """Test QuantumCircuit oracle"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover = Grover(oracle=oracle, good_state=["11"])
        grover_op = grover._grover_operator
        self.assertTrue(Operator(grover_op).equiv(Operator(self._expected_grover_op)))

    def test_oracle_statevector(self):
        """Test StateVector oracle"""
        mark_state = Statevector.from_label('11')
        grover = Grover(oracle=mark_state, good_state=['11'])
        grover_op = grover._grover_operator
        self.assertTrue(Operator(grover_op).equiv(Operator(self._expected_grover_op)))

    def test_state_preparation_quantumcircuit(self):
        """Test QuantumCircuit state_preparation"""
        state_preparation = QuantumCircuit(2)
        state_preparation.h(0)
        oracle = QuantumCircuit(3)
        oracle.cz(0, 1)
        grover = Grover(oracle=oracle, state_preparation=state_preparation,
                        good_state=["011"])
        grover_op = grover._grover_operator
        expected_grover_op = GroverOperator(oracle, state_preparation=state_preparation)
        self.assertTrue(Operator(grover_op).equiv(Operator(expected_grover_op)))

    def test_is_good_state_list(self):
        """Test List is_good_state"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        is_good_state = ["11", "00"]
        grover = Grover(oracle=oracle, good_state=is_good_state)
        self.assertListEqual(grover._is_good_state, ["11", "00"])

    def test_is_good_state_statevector(self):
        """Test StateVector is_good_state"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        is_good_state = Statevector.from_label('11')
        grover = Grover(oracle=oracle, good_state=is_good_state)
        self.assertTrue(grover._is_good_state.equiv(Statevector.from_label('11')))

    def test_grover_operator(self):
        """Test GroverOperator"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover_op = GroverOperator(oracle)
        grover = Grover(oracle=grover_op.oracle,
                        grover_operator=grover_op, good_state=["11"])
        grover_op = grover._grover_operator
        self.assertTrue(Operator(grover_op).equiv(Operator(self._expected_grover_op)))


class TestGroverPublicMethods(unittest.TestCase):
    """Test for the public methods of Grover"""

    def test_is_good_state(self):
        """Test is_good_state"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        list_str_good_state = ["11"]
        grover = Grover(oracle=oracle, good_state=list_str_good_state)
        self.assertTrue(grover.is_good_state("11"))

        statevector_good_state = Statevector.from_label('11')
        grover = Grover(oracle=oracle, good_state=statevector_good_state)
        self.assertTrue(grover.is_good_state("11"))


class TestGroverRun(unittest.TestCase):
    """Test running Grover's algorithm"""

    def test_grover_sv(self):
        """Test Grover with statevector simulator"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover = Grover(oracle=oracle, good_state=["11"], iterations=1)
        qi = QuantumInstance(StatevectorSimulator())
        result = grover.run(qi)
        self.assertEqual(result.top_measurement, '11')
        self.assertTrue(result.oracle_evaluation)

    def test_grover_qasm(self):
        """Test Grover with qasm simulator"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover = Grover(oracle=oracle, good_state=["11"], iterations=1)
        qi = QuantumInstance(AerSimulator(), shots=1000)
        result = grover.run(qi)
        self.assertEqual(result.top_measurement, '11')
        self.assertTrue(result.oracle_evaluation)


if __name__ == '__main__':
    unittest.main()
