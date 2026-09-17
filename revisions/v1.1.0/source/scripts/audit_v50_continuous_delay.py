#!/usr/bin/env python
"""Run the classifier-free V5 continuous signed-delay audit on Session T."""

from __future__ import annotations

import sys

from audit_v42_fold_evidence import main


if __name__ == "__main__":
    defaults = {
        "--protocol-config": "configs/experiments/v50_event_native_gate.yaml",
        "--spaces": "csd",
    }
    for flag, value in reversed(tuple(defaults.items())):
        if flag not in sys.argv:
            sys.argv[1:1] = [flag, value]
    if "--no-innovations" not in sys.argv:
        sys.argv.insert(1, "--no-innovations")
    if "--skip-injection" not in sys.argv:
        sys.argv.insert(1, "--skip-injection")
    main()
