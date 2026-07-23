""" Test Shor """

import unittest
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'programs'))

from qiskit_aer import AerSimulator
from shor import Shor
from quantum_instance import QuantumInstance


class TestShor(unittest.TestCase):
    """test Shor's algorithm"""

    def test_shor_factoring(self):
        """ shor factoring test """
        shor = Shor(15)
        result_dict = shor.run(QuantumInstance(AerSimulator(), shots=1000))
        self.assertListEqual(result_dict['factors'][0], [3, 5])
        self.assertTrue(result_dict["total_counts"] >= result_dict["successful_counts"])

    def test_shor_no_factors_5(self):
        """ shor no factors test for 5 """
        shor = Shor(5)
        qi = QuantumInstance(AerSimulator(), shots=1000)
        ret = shor.run(qi)
        self.assertTrue(ret['factors'] == [])
        self.assertTrue(ret["successful_counts"] == 0)

    def test_shor_no_factors_7(self):
        """ shor no factors test for 7 """
        shor = Shor(7)
        qi = QuantumInstance(AerSimulator(), shots=1000)
        ret = shor.run(qi)
        self.assertTrue(ret['factors'] == [])
        self.assertTrue(ret["successful_counts"] == 0)

    def test_shor_power_3_5(self):
        """ shor power test 3^5 """
        n_v = int(math.pow(3, 5))
        shor = Shor(n_v)
        qi = QuantumInstance(AerSimulator(), shots=1000)
        ret = shor.run(qi)
        self.assertTrue(ret['factors'] == [3])
        self.assertTrue(ret["total_counts"] >= ret["successful_counts"])

    def test_shor_power_5_3(self):
        """ shor power test 5^3 """
        n_v = int(math.pow(5, 3))
        shor = Shor(n_v)
        qi = QuantumInstance(AerSimulator(), shots=1000)
        ret = shor.run(qi)
        self.assertTrue(ret['factors'] == [5])
        self.assertTrue(ret["total_counts"] >= ret["successful_counts"])

    def test_shor_bad_input(self):
        """ shor bad input test """
        for n_v in [-1, 0, 1, 2, 4, 16]:
            with self.subTest(n_v=n_v):
                with self.assertRaises(ValueError):
                    Shor(n_v)

    def test_shor_modinv(self):
        """ shor modular inverse test """
        self.assertEqual(Shor.modinv(2, 15), 8)
        self.assertEqual(Shor.modinv(4, 15), 4)


if __name__ == '__main__':
    unittest.main()
