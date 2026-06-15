# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, Type
from navigation.SAC import MTransSAC
from navigation.DDPG import DDPG
from navigation.differentQ import MTransSACWithQ2
from navigation.SACwithoutGate import MTransSACWithoutGate
from navigation.SACwithoutTransformerAndGate import MTransSACWithoutTransformerAndGate
from navigation.baselinePointTransSAC import MHSASAC
from navigation.baselineSetTransSAC import SetTransSAC
from navigation.baselineDPRL import DPRLSAC
from navigation.baseNavigationAlgorithm import BaseNavigationAlgorithm
from navigation.differentQ import MTransSACWithQ2


def build_navigation_mappings(cfg) -> Tuple[Dict[str, BaseNavigationAlgorithm], Dict[str, str]]:
    """
    Build navigation algorithm instances and environment IDs from cfg.
    """
    navigation_algorithm = {
        "SAC": MTransSAC(cfg),
        "DDPG": DDPG(cfg),
        "SACWithQ2": MTransSACWithQ2(cfg),
        "SACWithoutGate": MTransSACWithoutGate(cfg),
        "SACWithoutTransformerAndGate": MTransSACWithoutTransformerAndGate(cfg),
        "PointTransSAC": MHSASAC(cfg),
        "SetTransSAC": SetTransSAC(cfg),
        "DPRL": DPRLSAC(cfg),
        "withQ2": MTransSACWithQ2(cfg),
    }
    environment = {
        "SAC": "UavAvoid-SAC",
        "DDPG": "UavAvoid-DDPG",
        "SACWithQ2": "UavAvoid-SAC",
        "SACWithoutGate": "UavAvoid-SAC",
        "SACWithoutTransformerAndGate": "UavAvoid-SAC",
        "PointTransSAC": "UavAvoid-SAC",
        "SetTransSAC": "UavAvoid-SAC",
        "DPRL": "UavAvoid-SAC",
        "withQ2": "UavAvoid-SAC",
    }
    return navigation_algorithm, environment


# SAC算法系列名称
SAC_ALGORITHM = ("SAC", "SACWithQ2", "SACWithoutGate", "SACWithoutTransformerAndGate", "PointTransSAC", "SetTransSAC", "DPRL", "withQ2")
# DDPG算法系列名称
DDPG_ALGORITHM = ("DDPG")
