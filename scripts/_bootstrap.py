"""Make the ``eeg_sl`` package importable when running scripts directly.

Each entry script does ``import _bootstrap`` first so you can run e.g.
``python scripts/train.py`` from anywhere without installing the package.
"""

import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
