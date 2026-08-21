"""
cost_core.gui - The desktop tool.

The window from the original lot cost model, with the added statistics
alongside it. Launch with ``python -m cost_core.gui`` or ``ce-core gui``.
"""

from cost_core.gui.app import LotCostApp, main

__all__ = ["LotCostApp", "main"]
