"""Compatibility alias for :mod:`mint_synth.core.config_v2`."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("mint_synth.core.config_v2")
