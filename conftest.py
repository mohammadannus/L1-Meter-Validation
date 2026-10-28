"""Pytest bootstrap — ensures the project root is on sys.path.

This lets tests import application modules (e.g. ``import api.app``)
regardless of the current working directory or how pytest is invoked.
"""

import sys
from pathlib import Path

# Insert the project root (this file's directory) at the front of sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))