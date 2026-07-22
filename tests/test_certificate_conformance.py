"""The certificate sign rule against its shared contract.

``tests/data/certificate_conformance.json`` pins, for a corpus of designs, the signed interaction graph
and every certificate the sign rule must produce. The same corpus is checked against the machine-checked
Lean reading in `grn-lean`, which consumes the designs :meth:`GRN.to_dict` emits, so the two readings
cannot drift apart unnoticed: a change to either that moves a verdict fails its own repository's test.

The corpus covers the cases where a sign rule can plausibly disagree with itself --- a ``hill2`` operator
that is non-monotone in an input, one whose input has no effect at all, a ``sum`` whose declared
``inputCount`` disagrees with how many species are wired to it, an operator whose basal and saturating
levels tie, and grounded kinetics whose decimals are not exactly representable in binary floating point.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from quiver.grn import GRN
from quiver.research.objective.certificate import (
    certifies,
    is_monotone,
    loop_signs,
    signed_interaction_graph,
)

CORPUS = json.loads((pathlib.Path(__file__).parent / "data" / "certificate_conformance.json").read_text())
CASES = CORPUS["cases"]


def _sorted_graph(edges) -> list:
    return sorted([list(edge) for edge in edges])


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_certificate_matches_contract(case: dict) -> None:
    grn = GRN.from_dict(case["design"])
    expected = case["expected"]

    assert _sorted_graph(signed_interaction_graph(grn)) == _sorted_graph(expected["signedInteractionGraph"])
    assert is_monotone(grn) is expected["monotone"]
    assert loop_signs(grn) == (expected["positiveLoop"], expected["negativeLoop"])
    for regime in ("sensor", "switch", "oscillator"):
        assert certifies(grn, regime) is expected["certifies"][regime]


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_design_round_trips_through_the_wire_schema(case: dict) -> None:
    """Every field the contract carries survives ``from_dict``/``to_dict``, so the design a checker reads
    is the design that was built --- including the Component grounding a verdict is attributed to."""
    design = case["design"]
    grn = GRN.from_dict(design)
    assert GRN.from_dict(grn.to_dict()) == grn
    assert [node.component_id for node in grn.nodes] == [node.get("componentId") for node in design["nodes"]]


def test_corpus_covers_the_divergence_prone_shapes() -> None:
    names = {case["name"] for case in CASES}
    assert {
        "hill2_xor_nonmonotone",
        "hill2_port1_inert",
        "sum_arity_under_declared",
        "tie_no_effect",
        "grounded_decimals",
    } <= names
