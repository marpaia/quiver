from __future__ import annotations

from quiver.component import apply_component, load_cello
from quiver.draw import to_mermaid
from quiver.generate.search import example_inducible
from quiver.grn import GRN, Edge, Node


def test_mermaid_renders_shapes_and_roles() -> None:
    mermaid = to_mermaid(example_inducible())
    assert mermaid.startswith("flowchart LR")
    assert '(("aTc"))' in mermaid, "the inducer is a circle"
    assert '(["GFP"])' in mermaid, "the reporter is a stadium"
    assert '[/"receiver"/]' in mermaid, "an operator is a slanted box"
    assert "senses" in mermaid and "expresses" in mermaid
    # example_inducible's inverter reads TetR and represses, so a dashed repression edge is drawn.
    assert "-. represses .->" in mermaid


def test_mermaid_shows_grounded_component_names() -> None:
    repressor = load_cello().repressors()[0]
    nodes = (
        Node("sup", "supplement", "aTc"),
        Node("op", "hill1", "op1", {"alpha": [100.0, 0.0], "K": 1.0, "n": 2.0}),
        Node("out", "reporter", "GFP", {"signal_id": "gfp"}),
        Node("reg", "regulator", "reg0"),
    )
    grn = GRN(nodes=nodes, edges=(Edge("sup", "op"), Edge("op", "out")))
    grounded = GRN(
        nodes=tuple(apply_component(n, repressor) if n.id == "op" else n for n in grn.nodes), edges=grn.edges
    )
    mermaid = to_mermaid(grounded)
    assert f'[/"{repressor.name}"/]' in mermaid, "a grounded operator shows its Component name"
