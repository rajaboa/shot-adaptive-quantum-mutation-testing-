"""Oracle utilities for BV, DJ, and Simon algorithms."""

import math
from qiskit import QuantumCircuit, QuantumRegister


class Oracle:
    """Simple oracle wrapper providing variable_register, output_register, and circuit."""

    def __init__(self, variable_register, output_register, circuit):
        self.variable_register = variable_register
        self.output_register = output_register
        self.circuit = circuit


def truth_table_oracle(bitmaps):
    """Build an oracle circuit from one or more bitmap strings.

    For a single bitmap string of length 2^n, builds f:{0,1}^n -> {0,1}.
    For a list of m bitmap strings, builds f:{0,1}^n -> {0,1}^m.

    Each bitmap[i] = '1' means f(i) = 1 (for single bitmap) or f_k(i) = 1.

    Args:
        bitmaps: A string or list of strings, each of length 2^n.

    Returns:
        Oracle with variable_register, output_register, and circuit.
    """
    if isinstance(bitmaps, str):
        bitmaps = [bitmaps]

    n = int(math.log2(len(bitmaps[0])))
    m = len(bitmaps)

    var_reg = QuantumRegister(n, 'v')
    out_reg = QuantumRegister(m, 'o')
    qc = QuantumCircuit(var_reg, out_reg)

    for k, bitmap in enumerate(bitmaps):
        for i, bit in enumerate(bitmap):
            if bit == '1':
                # Apply X gates to qubits that should be |0> for input i
                bin_i = format(i, f'0{n}b')
                for j in range(n):
                    if bin_i[j] == '0':
                        qc.x(var_reg[n - 1 - j])
                # Multi-controlled X targeting output qubit k
                if n == 1:
                    qc.cx(var_reg[0], out_reg[k])
                else:
                    qc.mcx(list(var_reg), out_reg[k])
                # Undo X gates
                for j in range(n):
                    if bin_i[j] == '0':
                        qc.x(var_reg[n - 1 - j])

    return Oracle(var_reg, out_reg, qc)
