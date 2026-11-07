"""Check resource settings before importing the frozen runner or JAX.

This append-only entry point is a post-collection correction. It does not
retroactively change the preregistered input script or create fresh seeds.
"""

import os
import runpy
import sys

from resource_policy import ResourcePolicyError, frozen_script, require_settings


def main() -> int:
    try:
        require_settings(os.environ, context="runner environment")
        script = frozen_script("run.py")
    except ResourcePolicyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    # Imports and allocations in the original module occur only after the guard.
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
