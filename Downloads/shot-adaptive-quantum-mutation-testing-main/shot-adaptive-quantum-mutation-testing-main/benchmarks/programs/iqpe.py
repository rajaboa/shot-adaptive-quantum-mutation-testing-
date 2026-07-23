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

"""The Iterative Quantum Phase Estimation Algorithm.

See https://arxiv.org/abs/quant-ph/0610214
"""

from typing import Optional, Dict, Any
import logging
import numpy as np

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter, LieTrotter

logger = logging.getLogger(__name__)


class IQPE:
    """The Iterative Quantum Phase Estimation algorithm.

    IQPE iteratively computes the phase so as to require fewer qubits.
    """

    def __init__(self,
                 operator: SparsePauliOp,
                 state_in: Optional[QuantumCircuit] = None,
                 num_time_slices: int = 1,
                 num_iterations: int = 1,
                 expansion_mode: str = 'suzuki',
                 expansion_order: int = 2,
                 shallow_circuit_concat: bool = False,
                 quantum_instance=None):
        if num_time_slices < 1:
            raise ValueError('num_time_slices must be >= 1')
        if num_iterations < 1:
            raise ValueError('num_iterations must be >= 1')
        if expansion_mode not in {'trotter', 'suzuki'}:
            raise ValueError("expansion_mode must be 'trotter' or 'suzuki'")
        if expansion_order < 1:
            raise ValueError('expansion_order must be >= 1')

        self._quantum_instance = quantum_instance
        self._state_in = state_in
        self._num_time_slices = num_time_slices
        self._num_iterations = num_iterations
        self._expansion_mode = expansion_mode
        self._expansion_order = expansion_order
        self._shallow_circuit_concat = shallow_circuit_concat
        self._state_register = None
        self._ancillary_register = None
        self._ancilla_phase_coef = None
        self._ret = {}  # type: Dict[str, Any]

        self._operator = operator
        self._num_qubits = operator.num_qubits

        # Compute translation and stretch for the operator
        # translation = sum of absolute coefficients
        self._ret['translation'] = float(np.sum(np.abs(operator.coeffs)))
        self._ret['stretch'] = 0.5 / self._ret['translation']

        # Build the translated operator: H' = H + translation * I
        identity_op = SparsePauliOp.from_list([('I' * self._num_qubits, self._ret['translation'])])
        self._translated_op = (operator + identity_op).simplify()

        # Build the stretched operator
        self._stretched_op = (self._ret['stretch'] * self._translated_op).simplify()

        # Extract identity coefficient for global phase
        self._ancilla_phase_coef = 0.0
        for pauli, coeff in zip(self._stretched_op.paulis, self._stretched_op.coeffs):
            label = str(pauli)
            if all(c == 'I' for c in label):
                self._ancilla_phase_coef = float(coeff.real)
                break

        # Build evolution operator (without identity term)
        non_identity_terms = []
        for pauli, coeff in zip(self._stretched_op.paulis, self._stretched_op.coeffs):
            label = str(pauli)
            if not all(c == 'I' for c in label):
                non_identity_terms.append((label, complex(coeff)))

        if non_identity_terms:
            self._evolution_op = SparsePauliOp.from_list(non_identity_terms)
        else:
            self._evolution_op = None

    def construct_circuit(self,
                          k: Optional[int] = None,
                          omega: float = 0,
                          measurement: bool = False) -> QuantumCircuit:
        """Construct the kth iteration Quantum Phase Estimation circuit."""
        if self._state_in is None:
            return None

        k = self._num_iterations if k is None else k
        a = QuantumRegister(1, name='a')
        q = QuantumRegister(self._num_qubits, name='q')
        self._ancillary_register = a
        self._state_register = q
        qc = QuantumCircuit(q)

        # Apply initial state
        qc.append(self._state_in, q)

        # hadamard on a[0]
        qc.add_register(a)
        qc.h(a[0])

        # controlled-U^(2^(k-1))
        if self._evolution_op is not None:
            power = 2 ** (k - 1)
            time = -2 * np.pi * self._num_time_slices
            if self._expansion_mode == 'suzuki':
                synthesis = SuzukiTrotter(order=self._expansion_order,
                                          reps=self._num_time_slices)
            else:
                synthesis = LieTrotter(reps=self._num_time_slices)

            evo_gate = PauliEvolutionGate(
                self._evolution_op,
                time=time * power,
                synthesis=synthesis
            )
            controlled_evo = evo_gate.control(1)
            qc.append(controlled_evo, [a[0]] + list(q))

        # global phase due to identity pauli
        qc.p(2 * np.pi * self._ancilla_phase_coef * (2 ** (k - 1)), a[0])

        # rz on a[0]
        qc.p(omega, a[0])

        # hadamard on a[0]
        qc.h(a[0])

        if measurement:
            c = ClassicalRegister(1, name='c')
            qc.add_register(c)
            qc.measure(self._ancillary_register, c)

        return qc

    def _estimate_phase_iteratively(self):
        """Iteratively estimate the phase."""
        self._ret['top_measurement_label'] = ''

        omega_coef = 0
        for k in range(self._num_iterations, 0, -1):
            omega_coef /= 2
            if self._quantum_instance.is_statevector:
                qc = self.construct_circuit(k, -2 * np.pi * omega_coef, measurement=False)
                result = self._quantum_instance.execute(qc)
                complete_state_vec = np.asarray(result.get_statevector(qc))
                ancilla_density_mat = _get_subsystem_density_matrix(
                    complete_state_vec,
                    range(self._num_qubits)
                )
                ancilla_density_mat_diag = np.diag(ancilla_density_mat)
                max_amplitude = max(ancilla_density_mat_diag.min(),
                                    ancilla_density_mat_diag.max(), key=abs)
                x = np.where(ancilla_density_mat_diag == max_amplitude)[0][0]
            else:
                qc = self.construct_circuit(k, -2 * np.pi * omega_coef, measurement=True)
                measurements = self._quantum_instance.execute(qc).get_counts(qc)

                if '0' not in measurements:
                    if '1' in measurements:
                        x = 1
                    else:
                        raise RuntimeError('Unexpected measurement {}.'.format(measurements))
                else:
                    if '1' not in measurements:
                        x = 0
                    else:
                        x = 1 if measurements['1'] > measurements['0'] else 0

            self._ret['top_measurement_label'] = \
                '{}{}'.format(x, self._ret['top_measurement_label'])
            omega_coef = omega_coef + x / 2
            logger.info('Reverse iteration %s of %s with measured bit %s',
                        k, self._num_iterations, x)
        return omega_coef

    def _compute_energy(self):
        """Compute the energy from phase estimation."""
        self._ret['phase'] = self._estimate_phase_iteratively()
        self._ret['top_measurement_decimal'] = sum([t[0] * t[1] for t in zip(
            [1 / 2 ** p for p in range(1, self._num_iterations + 1)],
            [int(n) for n in self._ret['top_measurement_label']]
        )])
        self._ret['energy'] = self._ret['phase'] / self._ret['stretch'] - self._ret['translation']

    def run(self, quantum_instance=None):
        """Run the algorithm."""
        if quantum_instance is not None:
            self._quantum_instance = quantum_instance
        return self._run()

    def _run(self):
        self._compute_energy()

        result = IQPEResult()
        result.translation = self._ret.get('translation')
        result.stretch = self._ret.get('stretch')
        result.top_measurement_label = self._ret.get('top_measurement_label')
        result.top_measurement_decimal = self._ret.get('top_measurement_decimal')
        result.eigenvalue = complex(self._ret.get('energy', 0))
        result.phase = self._ret.get('phase')
        return result


class IQPEResult:
    """IQPE Result."""

    def __init__(self):
        self.translation = None
        self.stretch = None
        self.top_measurement_label = None
        self.top_measurement_decimal = None
        self.eigenvalue = None
        self.phase = None

    def __getitem__(self, key):
        return getattr(self, key)


def _get_subsystem_density_matrix(statevector, trace_out_qubits):
    """Get reduced density matrix by tracing out specified qubits."""
    sv = Statevector(statevector)
    rho = partial_trace(sv, list(trace_out_qubits))
    return np.array(rho.data)
