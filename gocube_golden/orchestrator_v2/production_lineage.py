"""Topology-neutral name for the existing production lineage owner."""
from .torus9_production import Torus9ProductionLineage

# The implementation was already topology-neutral; retain the historical name
# as a compatibility alias while new orchestration composition uses this name.
ProductionLineage = Torus9ProductionLineage

__all__ = ["ProductionLineage", "Torus9ProductionLineage"]
