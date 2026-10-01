"""
sccg — Spin-Control Connectivity Graph framework.

Build resonance-windowed graphs over spin–orbit eigenstates and evaluate
endpoint-profile and pathway diagnostics D1–D5. Endpoint supports and shared
weights accompany the normalized coincidence; control fidelity is not inferred.
"""

from sccg.graph import SCCG
from sccg.io import load_manifold

__all__ = ["SCCG", "load_manifold"]
__version__ = "2.0.0"
