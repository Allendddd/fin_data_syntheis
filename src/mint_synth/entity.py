"""Compatibility alias for :mod:`mint_synth.synthesis.entity`."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("mint_synth.synthesis.entity")
