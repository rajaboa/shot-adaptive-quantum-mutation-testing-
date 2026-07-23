""" Test Simon """

import unittest
import math
import itertools
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'programs'))

import numpy as np
from qiskit_aer import AerSimulator, StatevectorSimulator
from simon import Simon
from quantum_instance import QuantumInstance
from oracle import truth_table_oracle

BITMAPS = [
    ['00011110', '01100110', '10101010'],
    ['10010110', '01010101', '10000010'],
    ['01101001', '10011001', '01100110'],
]
SIMULATORS = ['statevector_simulator', 'qasm_simulator']


class TestSimon(unittest.TestCase):
    """ Test Simon """

    def _get_backend(self, simulator):
        if simulator == 'statevector_simulator':
            return StatevectorSimulator()
        return AerSimulator()

    def test_simon(self):
        """ Simon test """
        for simon_input, simulator in itertools.product(BITMAPS, SIMULATORS):
            with self.subTest(simon_input=simon_input, simulator=simulator):
                # find the two keys that have matching values
                nbits = int(math.log(len(simon_input[0]), 2))
                vals = list(zip(*simon_input))[::-1]

                def find_pair():
                    for i, val in enumerate(vals):
                        for j in range(i + 1, len(vals)):
                            if val == vals[j]:
                                return i, j
                    return 0, 0

                k_1, k_2 = find_pair()
                hidden = np.binary_repr(k_1 ^ k_2, nbits)

                backend = self._get_backend(simulator)
                oracle = truth_table_oracle(simon_input)
                algorithm = Simon(oracle)
                quantum_instance = QuantumInstance(backend)
                result = algorithm.run(quantum_instance=quantum_instance)
                self.assertEqual(result['result'], hidden)


if __name__ == '__main__':
    unittest.main()
