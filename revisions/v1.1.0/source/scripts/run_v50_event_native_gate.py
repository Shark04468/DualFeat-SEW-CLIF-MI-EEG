#!/usr/bin/env python
"""Run the V5 continuous-delay, event-native SNN mechanism gate."""

from __future__ import annotations

import sys

from run_v40_delay_gate import main


if __name__ == "__main__":
    if "--protocol-config" not in sys.argv:
        sys.argv[1:1] = [
            "--protocol-config",
            "configs/experiments/v50_event_native_gate.yaml",
        ]
    main()
