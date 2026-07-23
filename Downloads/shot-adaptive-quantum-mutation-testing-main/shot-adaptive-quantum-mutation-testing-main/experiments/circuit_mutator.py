"""Circuit-level quantum mutation engine.

Generates mutants by directly modifying QuantumCircuit.data, bypassing
AST parsing, module compilation, and transpilation entirely.

Implements the same 5 mutation operators as QmutPy:
  - QGD: Quantum Gate Deletion
  - QGR: Quantum Gate Replacement
  - QGI: Quantum Gate Insertion (duplicate + replace second with equivalent)
  - QMD: Quantum Measurement Deletion
  - QMI: Quantum Measurement Insertion (X gate + measure before a gate)
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import getmembers, isfunction, signature
from copy import deepcopy

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import CircuitInstruction, Instruction


# ---------------------------------------------------------------------------
# Gate equivalence map (same logic as QmutPy)
# ---------------------------------------------------------------------------

_GATE_NAMES = [
    'ch', 'cp', 'cx', 'cy', 'cz', 'crx', 'cry', 'crz', 'ccx', 'cswap',
    'csx', 'cu', 'dcx', 'h', 'id', 'iswap',
    'ms', 'p', 'r', 'rx', 'rxx', 'ry', 'ryy', 'rz', 'rzx', 'rzz', 's',
    'sdg', 'swap', 'sx', 'x', 'y', 'z', 't', 'tdg', 'u',
]

_EQUIV_MAP: dict[str, list[str]] | None = None


def _build_equiv_map(backend=None) -> dict[str, list[str]]:
    """Build gate replacement map by (num_qubits, num_params).

    Unlike QmutPy's AST-level map (which groups by Python call arg count),
    this groups by actual quantum gate properties, avoiding incompetent mutations
    like replacing a 2-qubit gate with a 1-qubit gate.

    If backend is provided, only gates natively supported by the backend are
    included, avoiding incompetent mutants from unsupported gates.
    """
    global _EQUIV_MAP
    if _EQUIV_MAP is not None:
        return _EQUIV_MAP

    import inspect, math
    from qiskit.circuit import library as lib

    gate_classes = {
        'h': lib.HGate, 'x': lib.XGate, 'y': lib.YGate, 'z': lib.ZGate,
        's': lib.SGate, 'sdg': lib.SdgGate, 't': lib.TGate, 'tdg': lib.TdgGate,
        'sx': lib.SXGate, 'id': lib.IGate,
        'cx': lib.CXGate, 'cy': lib.CYGate, 'cz': lib.CZGate,
        'ch': lib.CHGate, 'csx': lib.CSXGate, 'dcx': lib.DCXGate,
        'swap': lib.SwapGate, 'iswap': lib.iSwapGate, 'cswap': lib.CSwapGate,
        'ccx': lib.CCXGate,
        'rx': lib.RXGate, 'ry': lib.RYGate, 'rz': lib.RZGate,
        'p': lib.PhaseGate,
        'crx': lib.CRXGate, 'cry': lib.CRYGate, 'crz': lib.CRZGate,
        'cp': lib.CPhaseGate,
        'rxx': lib.RXXGate, 'ryy': lib.RYYGate, 'rzx': lib.RZXGate,
        'rzz': lib.RZZGate,
        'r': lib.RGate,
    }

    # Filter to backend-supported gates if specified
    if backend is not None:
        try:
            basis = set(backend.configuration().basis_gates)
            gate_classes = {k: v for k, v in gate_classes.items() if k in basis}
        except Exception:
            pass

    # Group by (num_qubits, num_required_params)
    gate_key = {}
    for name, cls in gate_classes.items():
        try:
            sig = inspect.signature(cls.__init__)
            required = [p for p in sig.parameters.values()
                        if p.name != 'self' and p.default is inspect.Parameter.empty]
            nparams = len(required)
            if nparams == 0:
                g = cls()
            else:
                g = cls(*[math.pi / 4] * nparams)
            gate_key[name] = (g.num_qubits, nparams)
        except Exception:
            pass

    # Build map: each gate -> sorted list of replaceable gates
    groups: dict[tuple, list[str]] = {}
    for name, key in gate_key.items():
        groups.setdefault(key, []).append(name)

    _EQUIV_MAP = {}
    for name, key in gate_key.items():
        _EQUIV_MAP[name] = sorted(g for g in groups[key] if g != name)

    return _EQUIV_MAP


# ---------------------------------------------------------------------------
# Mutant descriptor
# ---------------------------------------------------------------------------

@dataclass
class CircuitMutant:
    """Describes a single circuit-level mutation."""
    index: int
    operator: str           # QGD, QGR, QGI, QMD, QMI
    target_index: int       # index into original circuit.data
    target_gate: str        # name of the original gate
    replacement_gate: str | None  # for QGR/QGI: the replacement gate name
    description: str        # human-readable description
    circuit: QuantumCircuit | None = None  # the mutant circuit (set by apply)


# ---------------------------------------------------------------------------
# Mutation generators
# ---------------------------------------------------------------------------

def _is_gate(inst: CircuitInstruction) -> bool:
    """True if instruction is a quantum gate (not barrier/measure/reset)."""
    return inst.operation.name not in ('barrier', 'measure', 'reset', 'delay')


def _is_measure(inst: CircuitInstruction) -> bool:
    return inst.operation.name == 'measure'


def generate_mutants(circuit: QuantumCircuit, backend=None) -> list[CircuitMutant]:
    """Generate all circuit-level mutants for the given circuit.

    Returns a list of CircuitMutant descriptors (without circuits yet).
    Call apply_mutant() to create the actual mutant circuit.

    If backend is provided, only generates replacement gates that the
    backend supports natively.
    """
    equiv = _build_equiv_map(backend)
    mutants = []
    idx = 0

    for i, inst in enumerate(circuit.data):
        gate_name = inst.operation.name

        # QGD: delete any quantum gate
        if _is_gate(inst):
            idx += 1
            mutants.append(CircuitMutant(
                index=idx, operator="QGD", target_index=i,
                target_gate=gate_name, replacement_gate=None,
                description=f"Delete {gate_name} at position {i}",
            ))

        # QGR: replace gate with each equivalent
        if _is_gate(inst) and gate_name in equiv:
            for repl in equiv[gate_name]:
                idx += 1
                mutants.append(CircuitMutant(
                    index=idx, operator="QGR", target_index=i,
                    target_gate=gate_name, replacement_gate=repl,
                    description=f"Replace {gate_name} -> {repl} at position {i}",
                ))

        # QGI: insert equivalent gate after existing gate (original + new)
        if _is_gate(inst) and gate_name in equiv:
            for repl in equiv[gate_name]:
                idx += 1
                mutants.append(CircuitMutant(
                    index=idx, operator="QGI", target_index=i,
                    target_gate=gate_name, replacement_gate=repl,
                    description=f"Insert {repl} after {gate_name} at position {i}",
                ))

        # QMD: delete measurement
        if _is_measure(inst):
            idx += 1
            mutants.append(CircuitMutant(
                index=idx, operator="QMD", target_index=i,
                target_gate="measure", replacement_gate=None,
                description=f"Delete measure at position {i}",
            ))

        # QMI: insert measurement before a gate (X + measure on target qubit)
        if _is_gate(inst) and len(inst.qubits) >= 1:
            idx += 1
            mutants.append(CircuitMutant(
                index=idx, operator="QMI", target_index=i,
                target_gate=gate_name, replacement_gate="measure",
                description=f"Insert measure before {gate_name} at position {i}",
            ))

    return mutants


def apply_mutant(original: QuantumCircuit, mutant: CircuitMutant) -> QuantumCircuit:
    """Create a mutant circuit by applying the mutation to a copy of the original.

    Returns the mutant QuantumCircuit, or raises ValueError if incompetent.
    """
    qc = original.copy()

    if mutant.operator == "QGD":
        # Delete the gate
        del qc.data[mutant.target_index]

    elif mutant.operator == "QGR":
        # Replace gate with equivalent
        old_inst = qc.data[mutant.target_index]
        new_gate = getattr(QuantumCircuit, mutant.replacement_gate)
        # Build replacement gate with same params
        try:
            gate_obj = _make_gate(mutant.replacement_gate, old_inst)
            qc.data[mutant.target_index] = CircuitInstruction(
                gate_obj, old_inst.qubits, old_inst.clbits
            )
        except Exception as e:
            raise ValueError(f"Incompetent: cannot replace {mutant.target_gate} "
                           f"with {mutant.replacement_gate}: {e}")

    elif mutant.operator == "QGI":
        # Insert new gate after the existing one (same qubits)
        old_inst = qc.data[mutant.target_index]
        try:
            gate_obj = _make_gate(mutant.replacement_gate, old_inst)
            new_inst = CircuitInstruction(gate_obj, old_inst.qubits, old_inst.clbits)
            qc.data.insert(mutant.target_index + 1, new_inst)
        except Exception as e:
            raise ValueError(f"Incompetent: cannot insert {mutant.replacement_gate}: {e}")

    elif mutant.operator == "QMD":
        # Delete measurement
        del qc.data[mutant.target_index]

    elif mutant.operator == "QMI":
        # Insert X gate + measurement before the target gate
        old_inst = qc.data[mutant.target_index]
        target_qubit = old_inst.qubits[0]

        # Add a new classical register for the inserted measurement
        cr = ClassicalRegister(1, name=f'qmi_{mutant.target_index}')
        qc.add_register(cr)

        # Insert X gate before target
        from qiskit.circuit.library import XGate
        x_inst = CircuitInstruction(XGate(), [target_qubit], [])
        qc.data.insert(mutant.target_index, x_inst)

        # Insert measure after X (which is now at target_index),
        # so measure goes at target_index + 1
        from qiskit.circuit.library import Measure
        m_inst = CircuitInstruction(Measure(), [target_qubit], [cr[0]])
        qc.data.insert(mutant.target_index + 1, m_inst)

    mutant.circuit = qc
    return qc


def _make_gate(gate_name: str, reference_inst: CircuitInstruction) -> Instruction:
    """Create a gate object compatible with the reference instruction.

    Uses the same parameters (angles) from the reference instruction's gate.
    """
    import math, inspect
    from qiskit.circuit import library as lib

    _GATE_CLASSES = {
        'h': lib.HGate, 'x': lib.XGate, 'y': lib.YGate, 'z': lib.ZGate,
        's': lib.SGate, 'sdg': lib.SdgGate, 't': lib.TGate, 'tdg': lib.TdgGate,
        'sx': lib.SXGate, 'id': lib.IGate,
        'cx': lib.CXGate, 'cy': lib.CYGate, 'cz': lib.CZGate,
        'ch': lib.CHGate, 'csx': lib.CSXGate, 'dcx': lib.DCXGate,
        'swap': lib.SwapGate, 'iswap': lib.iSwapGate, 'cswap': lib.CSwapGate,
        'ccx': lib.CCXGate,
        'rx': lib.RXGate, 'ry': lib.RYGate, 'rz': lib.RZGate,
        'p': lib.PhaseGate,
        'crx': lib.CRXGate, 'cry': lib.CRYGate, 'crz': lib.CRZGate,
        'cp': lib.CPhaseGate,
        'rxx': lib.RXXGate, 'ryy': lib.RYYGate, 'rzx': lib.RZXGate,
        'rzz': lib.RZZGate,
        'r': lib.RGate,
    }

    if gate_name not in _GATE_CLASSES:
        raise ValueError(f"Unknown gate: {gate_name}")

    gate_cls = _GATE_CLASSES[gate_name]

    # Count required positional params (excluding self and optional label/etc.)
    sig = inspect.signature(gate_cls.__init__)
    required = [p for p in sig.parameters.values()
                if p.name != 'self' and p.default is inspect.Parameter.empty]
    n_params = len(required)

    if n_params == 0:
        return gate_cls()

    # Reuse reference params if compatible
    ref_params = reference_inst.operation.params
    if ref_params and len(ref_params) >= n_params:
        return gate_cls(*ref_params[:n_params])

    # Fallback: use pi/4 for missing params
    return gate_cls(*[math.pi / 4] * n_params)
