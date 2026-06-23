"""The compositional grid DSL: programs, interpreter, and primitive registry."""

from arc_solver.dsl.program import Op, Program, serialize, deserialize, pretty
from arc_solver.dsl.interpreter import run, run_batch
from arc_solver.dsl.registry import load_primitives

# Populate the primitive registry on package import so the interpreter and
# enumerator always see a fully-loaded DSL. Idempotent (module import is cached).
load_primitives()

__all__ = ["Op", "Program", "serialize", "deserialize", "pretty", "run", "run_batch"]
