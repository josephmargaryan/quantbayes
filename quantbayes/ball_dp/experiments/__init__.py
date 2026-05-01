"""Experiment scripts and embedding loader modules.

The loader modules are intentionally not imported eagerly here because some of them
require heavyweight optional dependencies such as torchvision or transformers.  Import
the specific loader module needed for a dataset, or use the official run scripts.
"""

__all__: list[str] = []
