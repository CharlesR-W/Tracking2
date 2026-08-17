"""Compatibility imports for generic post-statistics test validators.

The obsolete artifact verifier CLI moved to
``scripts/deprecated/lw_post/verify_rank512_omnibus_battery.py``. The current
publication verifier is ``scripts/verify_lw_post_publication.py``.
"""

if __name__ == "__main__":
    raise SystemExit(
        "This obsolete verifier entry point has been retired. "
        "Use scripts/verify_lw_post_publication.py for the checked-in "
        "publication package, or the explicitly deprecated namespace for "
        "historical rank-512 artifacts."
    )

from scripts.deprecated.lw_post.verify_rank512_omnibus_battery import (  # noqa: E402
    require_common_relaxation_endpoint,
    require_distinct_checkpoint_hashes,
    require_exact_keys,
    require_valid_moment_diagnostics,
    require_valid_relaxation_records,
)

__all__ = [
    "require_common_relaxation_endpoint",
    "require_distinct_checkpoint_hashes",
    "require_exact_keys",
    "require_valid_moment_diagnostics",
    "require_valid_relaxation_records",
]
