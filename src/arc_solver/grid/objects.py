"""Object layer: object-model search + per-object attributes + helpers.

Extends the single fixed reading in :mod:`arc_solver.grid.components` (maximal
same-colour 4-connected regions) with a small, explicit search over *object
models* — the adjacency rule, whether an object may span colours, and which
colour is background — plus the attribute/placement helpers the object-centric
families (F1/F4) need.

Per ADR-009 this layer is used *inside* fitter-emitted primitives (e.g. the
``per_object`` combinator); the DSL pipeline's external contract stays
``Grid -> Grid`` and never exposes an object as a value type. See the GLOSSARY
entries 'Object' and 'Object model'.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arc_solver.grid.grid_ops import background_color

_NEIGH4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGH8 = _NEIGH4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass(frozen=True)
class ObjectModel:
    """How objects are read from a grid.

    Attributes:
        connectivity: 4 (orthogonal) or 8 (also diagonal) adjacency.
        multicolor: if True an object may span colours (any connected non-bg
            region); if False an object is a single-colour connected region.
        background: the colour treated as empty canvas and excluded from objects.
    """

    connectivity: int
    multicolor: bool
    background: int

    def to_spec(self) -> tuple[int, bool, int]:
        """A hashable, JSON-serialisable spec usable as an ``Op`` parameter."""
        return (self.connectivity, self.multicolor, self.background)

    @staticmethod
    def from_spec(spec: tuple) -> "ObjectModel":
        conn, multicolor, bg = spec
        return ObjectModel(int(conn), bool(multicolor), int(bg))


@dataclass(frozen=True)
class Obj:
    """A single object: its cells, the colours it covers, and its bounds."""

    cells: frozenset[tuple[int, int]]
    colors: frozenset[int]
    bbox: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col) inclusive

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def color(self) -> int | None:
        """The single colour of a mono object, or ``None`` if colour-spanning."""
        return next(iter(self.colors)) if len(self.colors) == 1 else None


def extract_objects(arr: np.ndarray, model: ObjectModel) -> list[Obj]:
    """Return the objects of ``arr`` under ``model`` (deterministic order).

    Objects are ordered top-to-bottom, left-to-right by bounding-box origin so
    downstream selection/placement is reproducible.
    """
    bg = model.background
    neigh = _NEIGH8 if model.connectivity == 8 else _NEIGH4
    h, w = arr.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[Obj] = []
    for i in range(h):
        for j in range(w):
            if seen[i, j] or arr[i, j] == bg:
                continue
            start_color = int(arr[i, j])
            stack = [(i, j)]
            seen[i, j] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in neigh:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc]:
                        continue
                    if arr[nr, nc] == bg:
                        continue
                    if model.multicolor or int(arr[nr, nc]) == start_color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            colors = frozenset(int(arr[r, c]) for r, c in cells)
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            out.append(
                Obj(
                    cells=frozenset(cells),
                    colors=colors,
                    bbox=(min(rs), min(cs), max(rs), max(cs)),
                )
            )
    out.sort(key=lambda o: (o.bbox[0], o.bbox[1]))
    return out


def candidate_models(arr: np.ndarray) -> list[ObjectModel]:
    """The small set of object readings a fitter searches, default reading first.

    ``{4, 8}-adjacency × {single-colour, colour-spanning} × background ∈
    {most-frequent, 0}``, deduplicated. The current default
    (4-adjacency, single-colour, most-frequent background) is placed first so an
    MDL tie breaks toward today's behaviour.
    """
    most_frequent = background_color(arr)
    backgrounds: list[int] = []
    for bg in (most_frequent, 0):
        if bg not in backgrounds:
            backgrounds.append(bg)

    models: list[ObjectModel] = []
    for bg in backgrounds:
        for multicolor in (False, True):
            for conn in (4, 8):
                model = ObjectModel(conn, multicolor, bg)
                if model not in models:
                    models.append(model)

    default = ObjectModel(4, False, most_frequent)
    if default in models:
        models.remove(default)
        models.insert(0, default)
    return models


def normalized_shape(obj: Obj) -> frozenset[tuple[int, int]]:
    """Translation-invariant cell set (relative to the bounding-box origin).

    Two objects with the same shape at different positions compare equal, so this
    is the key for shape-class reasoning.
    """
    r0, c0 = obj.bbox[0], obj.bbox[1]
    return frozenset((r - r0, c - c0) for (r, c) in obj.cells)
