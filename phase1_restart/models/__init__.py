"""Adapter registry. CellSAM is committed (models/cellsam.py) but NOT registered
because its weights are gated behind DeepCell auth (https://users.deepcell.org).
"""
from phase1_restart.models.mediar import MediarAdapter

REGISTRY = {
    "mediar": MediarAdapter,
}
