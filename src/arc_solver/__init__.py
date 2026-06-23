"""arc_solver — Test-time Bayesian program induction for ARC-AGI-2.

The package is organised into layers:

* ``io``     — task loading, validation, submission writing.
* ``grid``   — pure-numpy grid utilities, connected components, symmetry.
* ``dsl``    — the compositional domain-specific language and its interpreter.
* ``search`` — proposers (offline enumerative + LLM-amortised), MDL prior,
               verifier, refinement loop, and the solve engine.
* ``select`` — posterior weighting and the pass@2-aware top-2 selector.
* ``eval``   — evaluation harness and reporting.
* ``viz``    — grid rendering for figures and the media gallery.

The *offline* enumerative path (``search.enumerative``) is fully reproducible
on CPU and never touches the network. The LLM path is an optional second
proposer behind the same interface.
"""

__version__ = "0.1.0"
