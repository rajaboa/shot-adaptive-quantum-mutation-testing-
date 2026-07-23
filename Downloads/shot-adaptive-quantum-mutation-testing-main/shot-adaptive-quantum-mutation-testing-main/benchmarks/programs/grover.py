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

"""Grover's search algorithm."""

from typing import Optional, Union, List, Dict, Any, Callable
import logging
import operator
import math
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector, partial_trace

logger = logging.getLogger(__name__)


class Grover:
    r"""Grover's Search algorithm.

    Grover's Search is a quantum algorithm that can search through unstructured
    collections of records for particular targets with quadratic speedup.

    This implementation supports oracles as QuantumCircuit or Statevector.
    """

    def __init__(self,
                 oracle: Union[QuantumCircuit, Statevector],
                 good_state: Optional[Union[Callable[[str], bool],
                                            List[str], List[int], Statevector]] = None,
                 state_preparation: Optional[QuantumCircuit] = None,
                 iterations: Union[int, List[int]] = 1,
                 sample_from_iterations: bool = False,
                 post_processing: Optional[Callable] = None,
                 grover_operator: Optional[QuantumCircuit] = None,
                 quantum_instance=None):
        self._quantum_instance = quantum_instance
        self._oracle = oracle

        # Construct GroverOperator circuit
        if grover_operator is not None:
            self._grover_operator = grover_operator
        else:
            self._grover_operator = GroverOperator(oracle=oracle,
                                                   state_preparation=state_preparation)

        max_iterations = np.ceil(2 ** (len(self._grover_operator.reflection_qubits) / 2))

        if not isinstance(iterations, list):
            iterations = [iterations]

        # cutoff if max_iterations is exceeded
        self._iterations = []
        for iteration in iterations:
            self._iterations += [iteration]
            if iteration > max_iterations:
                break

        # check the type of good_state
        _check_is_good_state(good_state)

        self._is_good_state = good_state
        self._sample_from_iterations = sample_from_iterations
        self._post_processing = post_processing

        self._ret = {}  # type: Dict[str, Any]
        self.random = np.random.default_rng()

    @staticmethod
    def optimal_num_iterations(num_solutions: int, num_qubits: int) -> int:
        """Return the optimal number of iterations."""
        return math.floor(np.pi * np.sqrt(2 ** num_qubits / num_solutions) / 4)

    def _run_experiment(self, power):
        """Run a grover experiment for a given power of the Grover operator."""
        if self._quantum_instance.is_statevector:
            qc = self.construct_circuit(power, measurement=False)
            result = self._quantum_instance.execute(qc)
            statevector = np.asarray(result.get_statevector(qc))
            num_bits = len(self._grover_operator.reflection_qubits)
            # trace out work qubits
            if qc.width() != num_bits:
                rho = partial_trace(Statevector(statevector), range(num_bits, qc.width()))
                statevector = np.diag(rho.data)
            max_amplitude = max(statevector.max(), statevector.min(), key=abs)
            max_amplitude_idx = np.where(statevector == max_amplitude)[0][0]
            top_measurement = np.binary_repr(max_amplitude_idx, num_bits)

        else:
            qc = self.construct_circuit(power, measurement=True)
            measurement = self._quantum_instance.execute(qc).get_counts(qc)
            self._ret['measurement'] = measurement
            top_measurement = max(measurement.items(), key=operator.itemgetter(1))[0]

        self._ret['top_measurement'] = top_measurement

        return self.post_processing(top_measurement), self.is_good_state(top_measurement)

    def is_good_state(self, bitstr: str) -> bool:
        """Check whether a provided bitstring is a good state or not."""
        if callable(self._is_good_state):
            return self._is_good_state(bitstr)
        elif isinstance(self._is_good_state, list):
            if all(isinstance(good_bitstr, str) for good_bitstr in self._is_good_state):
                return bitstr in self._is_good_state
            else:
                return all(bitstr[good_index] == '1'
                           for good_index in self._is_good_state)
        # else isinstance(self._is_good_state, Statevector)
        return bitstr in self._is_good_state.probabilities_dict()

    def post_processing(self, measurement):
        """Do the post-processing to the measurement result."""
        if self._post_processing is not None:
            return self._post_processing(measurement)
        return measurement

    def construct_circuit(self, power: Optional[int] = None,
                          measurement: bool = False) -> QuantumCircuit:
        """Construct the circuit for Grover's algorithm with ``power`` Grover operators."""
        if power is None:
            power = self._iterations[0]

        qc = QuantumCircuit(self._grover_operator.num_qubits, name='Grover circuit')
        qc.compose(self._grover_operator.state_preparation, inplace=True)
        if power > 0:
            qc.compose(self._grover_operator.power(power), inplace=True)

        if measurement:
            measurement_cr = ClassicalRegister(len(self._grover_operator.reflection_qubits))
            qc.add_register(measurement_cr)
            qc.measure(self._grover_operator.reflection_qubits, measurement_cr)

        self._ret['circuit'] = qc
        return qc

    def run(self, quantum_instance=None) -> 'GroverResult':
        """Run the algorithm."""
        if quantum_instance is not None:
            self._quantum_instance = quantum_instance
        return self._run()

    def _run(self) -> 'GroverResult':
        for power in self._iterations:
            if self._sample_from_iterations:
                power = self.random.integers(power)
            assignment, oracle_evaluation = self._run_experiment(power)
            if oracle_evaluation:
                break

        result = GroverResult()
        if 'measurement' in self._ret:
            result.measurement = dict(self._ret['measurement'])
        if 'top_measurement' in self._ret:
            result.top_measurement = self._ret['top_measurement']
        if 'circuit' in self._ret:
            result.circuit = self._ret['circuit']
        result.assignment = assignment
        result.oracle_evaluation = oracle_evaluation
        return result

    @property
    def grover_operator(self) -> QuantumCircuit:
        """Returns grover_operator."""
        return self._grover_operator


class GroverResult:
    """Grover Result."""

    def __init__(self):
        self.measurement = None
        self.top_measurement = None
        self.circuit = None
        self.assignment = None
        self.oracle_evaluation = None

    def __getitem__(self, key):
        return getattr(self, key)


def _check_is_good_state(is_good_state):
    """Check whether a provided is_good_state is one of the supported types."""
    is_compatible = False
    if callable(is_good_state):
        is_compatible = True
    if isinstance(is_good_state, list):
        if all(isinstance(good_bitstr, str) for good_bitstr in is_good_state) or \
           all(isinstance(good_index, int) for good_index in is_good_state):
            is_compatible = True
    if isinstance(is_good_state, Statevector):
        is_compatible = True

    if not is_compatible:
        raise TypeError('Unsupported type "{}" of is_good_state'.format(type(is_good_state)))
