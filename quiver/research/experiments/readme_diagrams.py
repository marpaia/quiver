"""Regenerate the README's Mermaid diagrams as Component-grounded sensor cascades.

Builds a canonical linear inducible cascade of each requested depth and grounds every stage in a real SBOL
Component --- the input sensor from the corpus, each promoter stage in a distinct repressor gate
(orthogonally) --- then prints its ``quiver.to_mermaid`` rendering, so the README's example circuits show real
parts and are regenerated from the model rather than drawn by hand.

Run: ``uv run python quiver/research/experiments/readme_diagrams.py``
"""

from __future__ import annotations

import random

from quiver import to_mermaid
from quiver.component import ComponentLibrary, ground_operator, ground_sensor, load_cello
from quiver.generate.topology import sensor_cascade
from quiver.grn import GRN


def _distinct_repressors(library: ComponentLibrary, count: int) -> list:
    """``count`` repressor Components with pairwise-distinct regulators (orthogonality), preferring the
    named TetR-family protein repressors over the generically-named CRISPRi gates for a legible diagram."""
    order = sorted(library.repressors(), key=lambda c: (c.family != "TetR", c.id))
    chosen: list = []
    used: set[str] = set()
    for component in order:
        if component.regulator not in used:
            chosen.append(component)
            used.add(component.regulator)
        if len(chosen) == count:
            break
    return chosen


def grounded_cascade(library: ComponentLibrary, stages: int) -> GRN:
    """A linear ``stages``-promoter inducible cascade grounded in real Components."""
    grn = ground_sensor(sensor_cascade(stages + 1, random.Random(0)), library, "aTc")
    hill1_ids = [node.id for node in grn.nodes if node.kind == "hill1"]
    for node_id, component in zip(hill1_ids, _distinct_repressors(library, len(hill1_ids))):
        grn = ground_operator(grn, node_id, component)
    return grn


def main() -> None:
    library = load_cello()
    for stages in (2, 4):
        print(f"\n# {stages}-stage grounded cascade\n")
        print("```mermaid")
        print(to_mermaid(grounded_cascade(library, stages)))
        print("```")


if __name__ == "__main__":
    main()
