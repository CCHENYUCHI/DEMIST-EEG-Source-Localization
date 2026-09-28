"""Tiny YAML config loader shared by the entry scripts.

Usage in a script::

    from eeg_sl.config_io import load_config
    cfg = load_config()              # reads --config path from argv
    print(cfg["model"]["d_model"])

The config is returned as a plain nested dict.  A ``--set a.b=c`` style override
is supported for quick command-line tweaks without editing the YAML.
"""

import argparse
import ast

import yaml


def _coerce(value):
    """Best-effort convert a CLI string to int/float/bool/list, else keep str."""
    low = value.strip().lower()
    if low in ("true", "false"):          # accept YAML-style lowercase booleans
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _apply_override(cfg, dotted_key, value):
    keys = dotted_key.split('.')
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = _coerce(value)


def load_config(description="EEG source-localization pipeline"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    parser.add_argument("--set", nargs="*", default=[],
                        help="overrides, e.g. --set training.batch_size=8 model.K=200")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    for override in args.set:
        key, _, value = override.partition("=")
        _apply_override(cfg, key, value)

    return cfg
