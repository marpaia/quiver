from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("loica")

from quiver.learn.amortized import sample_conditioned, target_features, train_conditioned  # noqa: E402
from quiver.objective import TargetResponse  # noqa: E402


def test_target_features_encode_direction() -> None:
    up = target_features(TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="up"))
    down = target_features(TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="down"))
    assert len(up) == 3 and up[-1] == 1.0 and down[-1] == 0.0


@pytest.mark.sim
def test_conditioned_trains_and_samples_per_target() -> None:
    targets = [
        TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="up"),
        TargetResponse("GFP", ec50=10.0, dynamic_range=10.0, direction="down"),
    ]
    policy = train_conditioned(targets, steps=3, batch=4, max_steps=5, seed=0)
    for target in targets:
        terminals = sample_conditioned(policy, target, 4, max_steps=5, seed=1)
        assert len(terminals) == 4
        for grn in terminals:
            kinds = {n.kind for n in grn.nodes}
            assert "supplement" in kinds and "reporter" in kinds
