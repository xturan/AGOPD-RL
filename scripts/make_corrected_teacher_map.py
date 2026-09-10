#!/usr/bin/env python3
"""Compatibility entry point for the corrected Base--teacher map.

The former version hard-coded a GRPO student map. The report now defines the
diagnostic student as Qwen3-1.7B Base, so the canonical implementation lives in
``build_base_teacher_map.py`` and joins both remaps by prompt id.
"""

from build_base_teacher_map import main


if __name__ == "__main__":
    main()
