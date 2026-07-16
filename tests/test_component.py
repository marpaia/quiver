from __future__ import annotations

import random

from quiver.component import (
    Component,
    ComponentLibrary,
    apply_component,
    component_to_params,
    gate_availability,
    library_from_name,
    load_cello,
    merge,
)
from quiver.generate.realizable import is_realizable
from quiver.grn import GRN, Node, is_operator
from quiver.learn.gflownet import PolicyNet, _sample_trajectory


def _library() -> ComponentLibrary:
    return load_cello()


def _activator(cid: str, regulator: str, ymin: float = 0.05, ymax: float = 5.0) -> Component:
    return Component(
        id=cid,
        name=cid,
        role="activator",
        regulator=regulator,
        family="ECF",
        organism="Eco",
        ymax=ymax,
        ymin=ymin,
        K=None,
        n=None,
        dynamic_range=ymax / ymin,
        source="ecf:test",
        quality="literature",
    )


def test_cello_library_counts() -> None:
    library = _library()
    assert len(library.repressors()) == 63, "Cello corpus has 63 repressor gates"
    assert len(library.sensors()) == 21, "Cello corpus has 21 input sensors"
    # Every gate is a repressor; there are no activator Components in the corpus.
    assert all(c.role == "repressor" for c in library.repressors())
    assert all(c.role == "sensor" for c in library.sensors())


def test_repressors_carry_hill_kinetics_sensors_may_not() -> None:
    library = _library()
    assert all(c.K is not None and c.n is not None for c in library.repressors()), "gates carry K and n"
    assert all(c.K is None for c in library.sensors()), "Cello sensors carry no K"


def test_repressor_params_are_inverting_and_realizable() -> None:
    repressor = _library().repressors()[0]
    params = component_to_params(repressor)
    alpha = params["alpha"]
    assert isinstance(alpha, list) and alpha[0] > alpha[-1], "repressor is high-out at low-in (alpha=[ymax,ymin])"
    node = Node("op", "hill1", "op", params)
    assert is_realizable(GRN(nodes=(node,))), "grounded kinetics sit inside the realizable envelope"


def test_sensor_params_activate_with_defaulted_cooperativity() -> None:
    sensor = _library().sensors()[0]
    params = component_to_params(sensor)
    alpha = params["alpha"]
    assert isinstance(alpha, list) and alpha[-1] > alpha[0], "sensor is high-out at saturating inducer"
    assert params["K"] == 1.0 and params["n"] == 2.0, "sensor K/n default when uncharacterized"


def test_sensor_for_matches_inducer() -> None:
    library = _library()
    atc = library.sensor_for("aTc")
    assert atc is not None and "atc" in atc.name.lower(), "aTc resolves to an aTc sensor"
    assert library.sensor_for("no_such_molecule") is None


def test_apply_component_grounds_node_and_records_provenance() -> None:
    repressor = _library().repressors()[0]
    grounded = apply_component(Node("op1", "hill1", "op1"), repressor)
    assert grounded.component_id == repressor.id
    assert grounded.name == repressor.name
    assert grounded.params == component_to_params(repressor)


def test_orthogonality_key_is_the_regulator() -> None:
    # Distinct gate variants can share one transcription factor; grounding must not reuse a regulator.
    library = _library()
    regulators = [c.regulator for c in library.repressors()]
    assert len(set(regulators)) < len(regulators), "the corpus has multiple gate variants per regulator"


def test_library_reloads_by_name() -> None:
    assert load_cello().name == "cello"
    reloaded = library_from_name("cello")
    assert reloaded is not None and len(reloaded) == len(load_cello())
    assert library_from_name("") is None and library_from_name("unknown") is None


def test_mlp_grounded_rollout_builds_orthogonal_component_circuit() -> None:
    # The MLP grounded rollout is torch-only (no simulation): the feature-summary policy grounds every stage.
    library = load_cello()
    policy = PolicyNet(n_components=len(library.repressors()))
    grn, log_prob = _sample_trajectory(policy, random.Random(0), 6, library=library, inducer="aTc")

    operators = [n for n in grn.nodes if is_operator(n.kind)]
    assert operators and all(n.component_id is not None for n in operators), "every operator is grounded"
    regulators = [library.by_id(n.component_id).regulator for n in grn.nodes if n.component_id is not None]
    assert len(regulators) == len(set(regulators)), "no transcription factor is reused (orthogonality)"
    assert bool(log_prob.isfinite())


def test_component_id_round_trips_and_is_out_of_hashes() -> None:
    plain = Node("op1", "hill1", "op1", {"alpha": [100.0, 0.0], "K": 1.0, "n": 2.0})
    grounded = apply_component(plain, _library().repressors()[0])
    circuit = GRN(nodes=(grounded,), edges=())
    restored = GRN.from_dict(circuit.to_dict())
    assert restored.nodes[0].component_id == grounded.component_id, "component_id survives serialization"

    # component_id is provenance: two nodes differing only by it hash identically.
    a = Node("x", "regulator", "TF", {}, component_id="lib:one")
    b = Node("x", "regulator", "TF", {}, component_id="lib:two")
    ga, gb = GRN(nodes=(a,)), GRN(nodes=(b,))
    assert ga.structural_hash() == gb.structural_hash()
    assert ga.topology_hash() == gb.topology_hash()


def test_cello_components_are_measured_quality() -> None:
    assert all(c.quality == "measured" for c in load_cello().components)
    assert load_cello().repressors()[0].corpus == "cello"


def test_activator_params_activate_and_gates_include_both_roles() -> None:
    params = component_to_params(_activator("ECF01", "sigma1"))
    alpha = params["alpha"]
    assert isinstance(alpha, list) and alpha[-1] > alpha[0], "activator is high-out at high input (alpha=[ymin,ymax])"

    library = ComponentLibrary((_activator("ECF01", "sigma1"),) + load_cello().repressors()[:2], name="mix")
    gate_ids = {c.id for c in library.gates()}
    assert "ECF01" in gate_ids and len(library.gates()) == 3, "gates() spans repressors and activators"
    assert len(library.activators()) == 1


def test_merge_aggregates_and_dedups_with_provenance() -> None:
    cello = load_cello()
    activators = ComponentLibrary((_activator("ECF01", "sigma1"), _activator("ECF02", "sigma2")), name="ecf")
    combined = merge(cello, activators, name="cello+ecf")
    assert len(combined) == len(cello) + 2
    assert len(combined.activators()) == 2 and len(combined.repressors()) == len(cello.repressors())
    assert {c.corpus for c in combined.components} == {"cello", "ecf"}
    # De-dup by id: merging a library with itself does not double it.
    assert len(merge(activators, activators, name="x")) == 2


def test_gate_availability_masks_used_regulator_across_roles() -> None:
    activator = _activator("ECF01", "sigma1")
    library = ComponentLibrary((activator,) + load_cello().repressors()[:2], name="mix")
    # Grounding a node in the activator marks sigma1 used, so the activator becomes unavailable.
    node = apply_component(Node("op", "hill1", "op"), activator)
    grn = GRN(nodes=(node,))
    availability = gate_availability(grn, library)
    gates = library.gates()
    used = [g.regulator for g, ok in zip(gates, availability) if not ok]
    assert "sigma1" in used, "an already-used regulator is masked out regardless of role"


def test_library_from_name_merges_composite() -> None:
    # A composite name whose parts are all unknown returns None; a known single name resolves.
    assert library_from_name("cello") is not None
    assert library_from_name("cello+nope") is None
