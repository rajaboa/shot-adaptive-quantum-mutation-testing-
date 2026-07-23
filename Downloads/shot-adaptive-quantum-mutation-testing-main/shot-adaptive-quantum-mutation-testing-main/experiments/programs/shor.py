# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Shor's factoring algorithm."""

from typing import Optional, Union, Tuple, List
import math
import array
import fractions
import logging
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Instruction, ParameterVector
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector, partial_trace

logger = logging.getLogger(__name__)


def _is_power(n, return_decomposition=False):
    """Check if n is a perfect power. Returns (bool, base, power) if return_decomposition."""
    for p in range(2, int(math.log2(n)) + 1):
        base = round(n ** (1.0 / p))
        for b in [base - 1, base, base + 1]:
            if b > 1 and b ** p == n:
                if return_decomposition:
                    return True, b, p
                return True
    if return_decomposition:
        return False, 0, 0
    return False


class Shor:
    """Shor's factoring algorithm.

    Shor's Factoring algorithm finds the prime factors for input integer N
    in polynomial time.

    Adapted from https://github.com/ttlion/ShorAlgQiskit
    """

    def __init__(self, N: int = 15, a: int = 2, quantum_instance=None):
        if N < 3:
            raise ValueError('The input needs to be an odd integer greater than 2.')
        if a < 2:
            raise ValueError('The integer a needs to satisfy a >= 2.')
        if N < 1 or N % 2 == 0:
            raise ValueError('The input needs to be an odd integer greater than 1.')
        if a >= N or math.gcd(a, N) != 1:
            raise ValueError('The integer a needs to satisfy a < N and gcd(a, N) = 1.')

        self._quantum_instance = quantum_instance
        self._n = None
        self._up_qreg = None
        self._down_qreg = None
        self._aux_qreg = None
        self._N = N
        self._a = a
        self._ret = {"factors": [], "total_counts": 0, "successful_counts": 0}

        # check if the input integer is a power
        tf, b, p = _is_power(N, return_decomposition=True)
        if tf:
            logger.info('The input integer is a power: %s=%s^%s.', N, b, p)
            self._ret['factors'].append(b)

        self._qft = QFTGate(1)  # placeholder, will be resized
        self._iqft = QFTGate(1).inverse()

        self._phi_add_N = None
        self._iphi_add_N = None

    def _get_angles(self, a: int) -> np.ndarray:
        """Calculates the array of angles to be used in the addition in Fourier Space."""
        s = bin(int(a))[2:].zfill(self._n + 1)
        angles = np.zeros([self._n + 1])
        for i in range(0, self._n + 1):
            for j in range(i, self._n + 1):
                if s[j] == '1':
                    angles[self._n - i] += math.pow(2, -(j - i))
            angles[self._n - i] *= np.pi
        return angles[::-1]

    @staticmethod
    def _phi_add_gate(size: int, angles: Union[np.ndarray, ParameterVector]) -> Gate:
        """Gate that performs addition by a in Fourier Space."""
        circuit = QuantumCircuit(size, name="phi_add")
        for i, angle in enumerate(angles):
            circuit.p(angle, i)
        return circuit.to_gate()

    def _double_controlled_phi_add_mod_N(self, num_qubits, angles):
        """Creates a circuit which implements double-controlled modular addition by a."""
        circuit = QuantumCircuit(num_qubits, name="phi_add")

        ctl_up = 0
        ctl_down = 1
        ctl_aux = 2

        qubits = range(3, num_qubits)

        phi_add_a = self._phi_add_gate(len(qubits), angles)
        iphi_add_a = phi_add_a.inverse()

        circuit.append(phi_add_a.control(2), [ctl_up, ctl_down, *qubits])
        circuit.append(self._iphi_add_N, qubits)
        circuit.append(self._iqft, qubits)

        circuit.cx(qubits[0], ctl_aux)

        circuit.append(self._qft, qubits)
        circuit.append(self._phi_add_N, qubits)
        circuit.append(iphi_add_a.control(2), [ctl_up, ctl_down, *qubits])
        circuit.append(self._iqft, qubits)

        circuit.x(qubits[0])
        circuit.cx(qubits[0], ctl_aux)
        circuit.x(qubits[0])

        circuit.append(self._qft, qubits)
        circuit.append(phi_add_a.control(2), [ctl_up, ctl_down, *qubits])
        return circuit

    def _controlled_multiple_mod_N(self, num_qubits: int, a: int) -> Instruction:
        """Implements modular multiplication by a as an instruction."""
        circuit = QuantumCircuit(
            num_qubits, name="multiply_by_{}_mod_{}".format(a % self._N, self._N)
        )
        down = circuit.qubits[1: self._n + 1]
        aux = circuit.qubits[self._n + 1:]
        qubits = [aux[i] for i in reversed(range(self._n + 1))]
        ctl_up = 0
        ctl_aux = aux[-1]

        angle_params = ParameterVector("angles", length=len(aux) - 1)
        double_controlled_phi_add = self._double_controlled_phi_add_mod_N(
            len(aux) + 2, angle_params
        )
        idouble_controlled_phi_add = double_controlled_phi_add.inverse()

        circuit.append(self._qft, qubits)

        for i, ctl_down in enumerate(down):
            a_exp = (2 ** i) * a % self._N
            angles = self._get_angles(a_exp)
            bound = double_controlled_phi_add.assign_parameters({angle_params: angles})
            circuit.append(bound, [ctl_up, ctl_down, ctl_aux, *qubits])

        circuit.append(self._iqft, qubits)

        for j in range(self._n):
            circuit.cswap(ctl_up, down[j], aux[j])
        circuit.append(self._qft, qubits)

        a_inv = self.modinv(a, self._N)
        for i in reversed(range(len(down))):
            a_exp = (2 ** i) * a_inv % self._N
            angles = self._get_angles(a_exp)
            bound = idouble_controlled_phi_add.assign_parameters({angle_params: angles})
            circuit.append(bound, [ctl_up, down[i], ctl_aux, *qubits])

        circuit.append(self._iqft, qubits)
        return circuit.to_instruction()

    def construct_circuit(self, measurement: bool = False) -> QuantumCircuit:
        """Construct circuit."""
        self._n = math.ceil(math.log(self._N, 2))
        self._qft = QFTGate(self._n + 1)
        self._iqft = QFTGate(self._n + 1).inverse()

        self._up_qreg = QuantumRegister(2 * self._n, name='up')
        self._down_qreg = QuantumRegister(self._n, name='down')
        self._aux_qreg = QuantumRegister(self._n + 2, name='aux')

        circuit = QuantumCircuit(self._up_qreg, self._down_qreg, self._aux_qreg,
                                 name="Shor(N={}, a={})".format(self._N, self._a))

        self._phi_add_N = self._phi_add_gate(self._aux_qreg.size - 1, self._get_angles(self._N))
        self._iphi_add_N = self._phi_add_N.inverse()

        circuit.h(self._up_qreg)
        circuit.x(self._down_qreg[0])

        for i, ctl_up in enumerate(self._up_qreg):
            a = int(pow(self._a, pow(2, i)))
            controlled_multiple_mod_N = self._controlled_multiple_mod_N(
                len(self._down_qreg) + len(self._aux_qreg) + 1, a,
            )
            circuit.append(
                controlled_multiple_mod_N, [ctl_up, *self._down_qreg, *self._aux_qreg]
            )

        # Apply inverse QFT
        iqft = QFTGate(len(self._up_qreg)).inverse()
        circuit.append(iqft, self._up_qreg)

        if measurement:
            up_cqreg = ClassicalRegister(2 * self._n, name='m')
            circuit.add_register(up_cqreg)
            circuit.measure(self._up_qreg, up_cqreg)

        return circuit

    @staticmethod
    def modinv(a: int, m: int) -> int:
        """Returns the modular multiplicative inverse of a with respect to the modulus m."""
        def egcd(a: int, b: int) -> Tuple[int, int, int]:
            if a == 0:
                return b, 0, 1
            else:
                g, y, x = egcd(b % a, a)
                return g, x - (b // a) * y, y

        g, x, _ = egcd(a, m)
        if g != 1:
            raise ValueError("The greatest common divisor of {} and {} is {}, so the "
                             "modular inverse does not exist.".format(a, m, g))
        return x % m

    def _get_factors(self, measurement: str) -> Optional[List[int]]:
        """Apply the continued fractions to find r and the gcd to find the desired factors."""
        x_final = int(measurement, 2)
        logger.info('In decimal, x_final value for this result is: %s.', x_final)

        if x_final <= 0:
            fail_reason = 'x_final value is <= 0, there are no continued fractions.'
        else:
            fail_reason = None
            logger.debug('Running continued fractions for this case.')

        T_upper = len(measurement)
        T = pow(2, T_upper)
        x_over_T = x_final / T

        i = 0
        b = array.array('i')
        t = array.array('f')

        b.append(math.floor(x_over_T))
        t.append(x_over_T - b[i])

        exponential = 0.0
        while i < self._N and fail_reason is None:
            if i > 0:
                b.append(math.floor(1 / t[i - 1]))
                t.append((1 / t[i - 1]) - b[i])

            denominator = self._calculate_continued_fraction(b)
            i += 1

            if denominator % 2 == 1:
                logger.debug('Odd denominator, will try next iteration of continued fractions.')
                continue

            if denominator < 1000:
                exponential = pow(self._a, denominator / 2)

            if exponential > 1000000000:
                fail_reason = 'denominator of continued fraction is too big.'
            else:
                putting_plus = int(exponential + 1)
                putting_minus = int(exponential - 1)
                one_factor = math.gcd(putting_plus, self._N)
                other_factor = math.gcd(putting_minus, self._N)

                if any(factor in {1, self._N} for factor in (one_factor, other_factor)):
                    logger.debug('Found just trivial factors, not good enough.')
                    if t[i - 1] == 0:
                        fail_reason = 'the continued fractions found exactly x_final/(2^(2n)).'
                else:
                    return sorted((one_factor, other_factor))

        logger.debug(
            'Cannot find factors from measurement %s because %s',
            measurement, fail_reason or 'it took too many attempts.'
        )
        return None

    @staticmethod
    def _calculate_continued_fraction(b: array.array) -> int:
        """Calculate the continued fraction of x/T from the current terms of expansion b."""
        x_over_T = 0

        for i in reversed(range(len(b) - 1)):
            x_over_T = 1 / (b[i + 1] + x_over_T)

        x_over_T += b[0]

        frac = fractions.Fraction(x_over_T).limit_denominator()

        logger.debug('Approximation number %s of continued fractions:', len(b))
        logger.debug("Numerator:%s \t\t Denominator: %s.", frac.numerator, frac.denominator)
        return frac.denominator

    def run(self, quantum_instance=None):
        """Run the algorithm."""
        if quantum_instance is not None:
            self._quantum_instance = quantum_instance
        return self._run()

    def _run(self):
        if not self._ret['factors']:
            logger.debug('Running with N=%s and a=%s.', self._N, self._a)

            if self._quantum_instance.is_statevector:
                circuit = self.construct_circuit(measurement=False)
                logger.warning('The statevector_simulator might lead to '
                               'subsequent computation using too much memory.')
                result = self._quantum_instance.execute(circuit)
                complete_state_vec = np.asarray(result.get_statevector(circuit))
                up_qreg_density_mat = _get_subsystem_density_matrix(
                    complete_state_vec,
                    range(2 * self._n, 4 * self._n + 2)
                )
                up_qreg_density_mat_diag = np.diag(up_qreg_density_mat)

                counts = {}
                for i, v in enumerate(up_qreg_density_mat_diag):
                    if not v == 0:
                        counts[bin(int(i))[2:].zfill(2 * self._n)] = v ** 2
            else:
                circuit = self.construct_circuit(measurement=True)
                counts = self._quantum_instance.execute(circuit).get_counts(circuit)

            self._ret["total_counts"] = len(counts)

            for measurement in list(counts.keys()):
                logger.info("------> Analyzing result %s.", measurement)
                factors = self._get_factors(measurement)

                if factors:
                    logger.info(
                        'Found factors %s from measurement %s.',
                        factors, measurement
                    )
                    self._ret["successful_counts"] += 1
                    if factors not in self._ret['factors']:
                        self._ret['factors'].append(factors)

        return self._ret


def _get_subsystem_density_matrix(statevector, trace_out_qubits):
    """Get reduced density matrix by tracing out specified qubits."""
    sv = Statevector(statevector)
    rho = partial_trace(sv, list(trace_out_qubits))
    return np.array(rho.data)
