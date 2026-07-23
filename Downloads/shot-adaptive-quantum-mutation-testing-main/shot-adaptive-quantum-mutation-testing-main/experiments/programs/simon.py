# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Simon's algorithm.
"""

from typing import Dict, Any
import numpy as np
from sympy import Matrix, mod_inverse

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace


class Simon:
    r"""
    The Simon algorithm.

    The Simon algorithm finds a hidden integer s in {0,1}^n from an oracle f_s
    that satisfies f_s(x) = f_s(y) iff y = x xor s.
    """

    def __init__(self, oracle, quantum_instance=None):
        self._oracle = oracle
        self._quantum_instance = quantum_instance
        self._circuit = None
        self._ret = {}  # type: Dict[str, Any]

    def construct_circuit(self, measurement=False):
        if self._circuit is not None:
            return self._circuit

        oracle = self._oracle.circuit
        self._circuit = QuantumCircuit(*oracle.qregs)

        # preoracle hadamard gates
        self._circuit.h(self._oracle.variable_register)

        # apply oracle
        self._circuit.compose(oracle, inplace=True)

        # postoracle hadamard gates
        self._circuit.h(self._oracle.variable_register)

        # measurement circuit
        if measurement:
            measurement_cr = ClassicalRegister(len(self._oracle.variable_register), name='m')
            self._circuit.add_register(measurement_cr)
            self._circuit.measure(self._oracle.variable_register, measurement_cr)

        return self._circuit

    def _interpret_measurement(self, measurements):
        # reverse measurement bitstrings and remove all zero entry
        linear = [(k[::-1], v) for k, v in measurements.items()
                  if k != "0" * len(self._oracle.variable_register)]
        # sort bitstrings by their probabilities
        linear.sort(key=lambda x: x[1], reverse=True)

        # construct matrix
        equations = []
        for k, _ in linear:
            equations.append([int(c) for c in k])
        y = Matrix(equations)

        # perform gaussian elimination
        y_transformed = y.rref(iszerofunc=lambda x: x % 2 == 0)

        def mod(x, modulus):
            numer, denom = x.as_numer_denom()
            return numer * mod_inverse(denom, modulus) % modulus
        y_new = y_transformed[0].applyfunc(lambda x: mod(x, 2))

        # determine hidden string from final matrix
        rows, _ = y_new.shape
        hidden = [0] * len(self._oracle.variable_register)
        for r in range(rows):
            yi = [i for i, v in enumerate(list(y_new[r, :])) if v == 1]
            if len(yi) == 2:
                hidden[yi[0]] = '1'
                hidden[yi[1]] = '1'
        return "".join(str(x) for x in hidden)[::-1]

    def run(self, quantum_instance=None):
        if quantum_instance is not None:
            self._quantum_instance = quantum_instance
        self._circuit = None
        return self._run()

    def _run(self):
        if self._quantum_instance.is_statevector:
            qc = self.construct_circuit(measurement=False)
            result = self._quantum_instance.execute(qc)
            complete_state_vec = result.get_statevector(qc)
            variable_register_density_matrix = _get_subsystem_density_matrix(
                complete_state_vec,
                range(len(self._oracle.variable_register), qc.width())
            )
            variable_register_density_matrix_diag = np.diag(variable_register_density_matrix)
            measurements = {
                np.binary_repr(idx, width=len(self._oracle.variable_register)):
                    abs(variable_register_density_matrix_diag[idx]) ** 2
                for idx in range(len(variable_register_density_matrix_diag))
                if abs(variable_register_density_matrix_diag[idx]) > 1e-8
            }
        else:
            qc = self.construct_circuit(measurement=True)
            measurements = self._quantum_instance.execute(qc).get_counts(qc)

        self._ret['result'] = self._interpret_measurement(measurements)
        return self._ret


def _get_subsystem_density_matrix(statevector, trace_out_qubits):
    """Get reduced density matrix by tracing out specified qubits."""
    sv = Statevector(statevector)
    rho = partial_trace(sv, list(trace_out_qubits))
    return np.array(rho.data)
