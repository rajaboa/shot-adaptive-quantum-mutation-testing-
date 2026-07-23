"""MQTBench bridge: forwards 'pricingput' to adaptive_runner.py.
Do not modify -- auto-generated for circuit-level mutation testing.
"""
import sys
import os

_MQT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                 "experimentMQTBench")
)
if _MQT not in sys.path:
    sys.path.insert(0, _MQT)

from programs import REGISTRY as _REGISTRY


def build_circuit(n_qubits, measurement=True, **kwargs):
    return _REGISTRY['pricingput'].build_circuit(n_qubits)
