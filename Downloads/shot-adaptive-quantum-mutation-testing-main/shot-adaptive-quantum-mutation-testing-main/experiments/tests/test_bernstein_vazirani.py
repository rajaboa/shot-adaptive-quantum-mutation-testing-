""" Test Bernstein Vazirani """

import unittest
import itertools
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'programs'))

from qiskit_aer import AerSimulator, StatevectorSimulator
from bernstein_vazirani import BernsteinVazirani
from quantum_instance import QuantumInstance
from oracle import truth_table_oracle

BITMAPS = ['00111100', '01011010']
SIMULATORS = ['statevector_simulator', 'qasm_simulator']


class TestBernsteinVazirani(unittest.TestCase):
    """ Test Bernstein Vazirani """

    def _get_backend(self, simulator):
        if simulator == 'statevector_simulator':
            return StatevectorSimulator()
        return AerSimulator()

    def test_bernstein_vazirani(self):
        """ Test BV with different bitmaps and simulators """
        for bv_input, simulator in itertools.product(BITMAPS, SIMULATORS):
            with self.subTest(bv_input=bv_input, simulator=simulator):
                nbits = int(math.log(len(bv_input), 2))
                # compute the ground-truth classically
                parameter = ""
                for i in reversed(range(nbits)):
                    bit = bv_input[2 ** i]
                    parameter += bit

                backend = self._get_backend(simulator)
                oracle = truth_table_oracle(bv_input)
                algorithm = BernsteinVazirani(oracle)
                quantum_instance = QuantumInstance(backend)
                result = algorithm.run(quantum_instance=quantum_instance)
                self.assertEqual(result['result'], parameter)


if __name__ == '__main__':
    unittest.main()
