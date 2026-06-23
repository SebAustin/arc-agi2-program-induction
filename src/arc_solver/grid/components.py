"""Connected-component extraction (the 'objectness' core-knowledge prior).

Objects are maximal connected regions of equal colour (excluding the background),
under 4- or 8-connectivity. This is implemented with a small flood-fill rather
than a SciPy dependency to keep the package dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arc_solver.grid.grid_ops import background_color


@dataclass(frozen=True)
class Component:
    """A connected region of a single colour.

    Attributes:
        color: the colour of the region.
        cells: frozenset of (row, col) coordinates in the original grid.
        bbox: (min_row, min_col, max_row, max_col) inclusive bounds.
    """

    color: int
    cells: frozenset[tuple[int, int]]
    bbox: tuple[int, int, int, int]

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1


_NEIGH4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGH8 = _NEIGH4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def connected_components(
    arr: np.ndarray, *, connectivity: int = 4, background: int | None = None
) -> list[Component]:
    """Return colour-homogeneous connected components, excluding the background.

    Args:
        arr: 2-D int array.
        connectivity: 4 or 8.
        background: colour treated as empty; defaults to the most common colour.
    """
    if background is None:
        background = background_color(arr)
    neigh = _NEIGH8 if connectivity == 8 else _NEIGH4
    h, w = arr.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[Component] = []
    for i in range(h):
        for j in range(w):
            if seen[i, j] or arr[i, j] == background:
                continue
            color = int(arr[i, j])
            stack = [(i, j)]
            seen[i, j] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in neigh:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and arr[nr, nc] == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            out.append(
                Component(
                    color=color,
                    cells=frozenset(cells),
                    bbox=(min(rs), min(cs), max(rs), max(cs)),
                )
            )
    # Deterministic ordering: top-to-bottom, left-to-right by bbox origin.
    out.sort(key=lambda comp: (comp.bbox[0], comp.bbox[1], comp.color))
    return out
