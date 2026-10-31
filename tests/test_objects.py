"""Tests for the object layer (grid/objects.py) and the per_object combinator.

See ADR-009 and the GLOSSARY entries 'Object' / 'Object model'. The object layer
is used *inside* fitter-emitted primitives; the pipeline stays Grid -> Grid.
"""

from __future__ import annotations

import numpy as np

from arc_solver.grid.objects import (
    ObjectModel,
    candidate_models,
    extract_objects,
    normalized_shape,
)


def test_object_model_spec_roundtrip():
    # Arrange
    model = ObjectModel(connectivity=8, multicolor=True, background=0)
    # Act
    restored = ObjectModel.from_spec(model.to_spec())
    # Assert
    assert restored == model
    assert hash(restored) == hash(model)  # usable as a hashable Op param


def test_extract_mono_two_components():
    arr = np.array([[1, 1, 0, 2], [1, 0, 0, 2], [0, 0, 0, 0]])
    objs = extract_objects(arr, ObjectModel(4, False, 0))
    assert len(objs) == 2
    assert sorted(o.size for o in objs) == [2, 3]


def test_connectivity_4_vs_8_diagonal():
    arr = np.array([[1, 0], [0, 1]])  # two diagonally-touching cells, background 0
    assert len(extract_objects(arr, ObjectModel(4, False, 0))) == 2
    assert len(extract_objects(arr, ObjectModel(8, False, 0))) == 1


def test_multicolor_merges_adjacent_colours():
    arr = np.array([[1, 2, 0], [0, 0, 0]])  # colours 1 and 2 orthogonally adjacent
    mono = extract_objects(arr, ObjectModel(4, False, 0))
    poly = extract_objects(arr, ObjectModel(4, True, 0))
    assert len(mono) == 2
    assert len(poly) == 1
    assert poly[0].colors == frozenset({1, 2})
    assert poly[0].color is None  # colour-spanning object has no single colour


def test_single_colour_object_reports_attributes():
    arr = np.array([[3, 3], [0, 0]])
    (obj,) = extract_objects(arr, ObjectModel(4, False, 0))
    assert obj.color == 3
    assert obj.colors == frozenset({3})
    assert obj.bbox == (0, 0, 0, 1)
    assert (obj.height, obj.width, obj.size) == (1, 2, 2)


def test_background_choice_changes_objects():
    arr = np.array([[0, 0, 5], [0, 0, 5]])  # most-frequent colour is 0
    bg0 = extract_objects(arr, ObjectModel(4, False, 0))
    bg5 = extract_objects(arr, ObjectModel(4, False, 5))
    assert len(bg0) == 1 and bg0[0].color == 5
    assert len(bg5) == 1 and bg5[0].color == 0


def test_candidate_models_default_first_and_deduped():
    arr = np.array([[0, 0, 1], [0, 0, 1]])  # most-frequent background = 0
    models = candidate_models(arr)
    assert models[0] == ObjectModel(4, False, 0)  # the current default, first
    assert len(models) == len(set(models))  # no duplicates
    assert any(m.connectivity == 8 for m in models)
    assert any(m.multicolor for m in models)


def test_normalized_shape_translation_invariant():
    a = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
    b = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]])
    (oa,) = extract_objects(a, ObjectModel(4, False, 0))
    (ob,) = extract_objects(b, ObjectModel(4, False, 0))
    assert normalized_shape(oa) == normalized_shape(ob)


def test_per_object_applies_inner_in_place():
    from arc_solver.dsl.primitives_object import per_object
    from arc_solver.dsl.registry import load_primitives

    load_primitives()
    arr = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
    # Flip each object's bounding-box block horizontally, in place.
    out = per_object(arr, "flip_h", (), (4, False, 0))
    assert out is not None
    assert out.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]


def test_per_object_none_when_no_objects():
    from arc_solver.dsl.primitives_object import per_object

    arr = np.zeros((3, 3), dtype=int)
    assert per_object(arr, "flip_h", (), (4, False, 0)) is None


def test_per_object_fitter_only_not_enumerated():
    # It must contribute zero grounded ops so the enumerative search is untouched.
    from arc_solver.dsl.registry import PRIMITIVES, TaskContext, load_primitives

    load_primitives()
    assert "per_object" in PRIMITIVES
    ctx = TaskContext(colors=(0, 1), input_colors=(0, 1), scale_factors=(), same_shape=True)
    assert list(PRIMITIVES["per_object"].param_space(ctx)) == []
