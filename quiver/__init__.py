"""quiver: machine learning for the diverse design of gene-regulatory circuits.

Describe what you want a cell to do and :func:`design` returns a batch of candidate circuits built from
parts you have characterized. The behavior is four orthogonal choices --- :class:`Responds`,
:class:`Logic`, ``memory`` and :class:`Timing` --- and never a number: quantities are measured and reported
rather than supplied.
"""

from __future__ import annotations

from quiver.component import Component, ComponentLibrary
from quiver.draw import to_mermaid
from quiver.grn import GRN, NODE_SPECS, Edge, Node, NodeKind, ParamValue, default_params, is_operator
from quiver.intent import Capabilities, Contradiction, Intent, Logic, NotSupportedYet, Prefer, Reads, Responds, Timing
from quiver.parts import (
    DifferentConditions,
    Inducer,
    Measurement,
    NoReporterCalled,
    NoSensorFor,
    NotCharacterized,
    Organism,
    Parts,
    Provenance,
    Reporter,
    Strain,
    combine,
    parts,
)
from quiver.portfolio import Designer, Portfolio, design, train
from quiver.report import Quantity, Report

__all__ = [
    "Capabilities",
    "Designer",
    "Component",
    "ComponentLibrary",
    "Contradiction",
    "DifferentConditions",
    "Edge",
    "GRN",
    "Inducer",
    "Intent",
    "Logic",
    "Measurement",
    "NODE_SPECS",
    "NoReporterCalled",
    "NoSensorFor",
    "Node",
    "NodeKind",
    "NotCharacterized",
    "NotSupportedYet",
    "Organism",
    "ParamValue",
    "Parts",
    "Portfolio",
    "Prefer",
    "Provenance",
    "Quantity",
    "Reads",
    "Report",
    "Reporter",
    "Responds",
    "Strain",
    "Timing",
    "combine",
    "default_params",
    "design",
    "is_operator",
    "parts",
    "to_mermaid",
    "train",
]
