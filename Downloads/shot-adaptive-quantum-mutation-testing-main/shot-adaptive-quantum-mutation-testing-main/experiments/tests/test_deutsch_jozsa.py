""" Test Deutsch Jozsa """

import unittest
import itertools
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'programs'))

from qiskit_aer import AerSimulator, StatevectorSimulator
from deutsch_jozsa import DeutschJozsa
from quantum_instance import QuantumInstance
from oracle import truth_table_oracle

BITMAPS = ['0000', '0101', '1111', '11110000']
SIMULATORS = ['statevector_simulator', 'qasm_simulator']


class TestDeutschJozsa(unittest.TestCase):
    """ Test Deutsch Jozsa """

    def _get_backend(self, simulator):
        if simulator == 'statevector_simulator':
            return StatevectorSimulator()
        return AerSimulator()

    def test_deutsch_jozsa(self):
        """ Deutsch Jozsa test """
        for dj_input, simulator in itertools.product(BITMAPS, SIMULATORS):
            with self.subTest(dj_input=dj_input, simulator=simulator):
                backend = self._get_backend(simulator)
                oracle = truth_table_oracle(dj_input)
                algorithm = DeutschJozsa(oracle)
                quantum_instance = QuantumInstance(backend)
                result = algorithm.run(quantum_instance=quantum_instance)
                if sum([int(i) for i in dj_input]) == len(dj_input) / 2:
                    self.assertTrue(result['result'] == 'balanced')
                else:
                    self.assertTrue(result['result'] == 'constant')


if __name__ == '__main__':
    unittest.main()
