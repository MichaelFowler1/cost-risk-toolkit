"""Shared fixtures for the lot cost engine tests.

These mirror the desktop tool's bundled example lots, so the engine tests
copied from that tool run against the same data here."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def analogy_df() -> pd.DataFrame:
    """The desktop tool's bundled example lots, in the shape
    run_lot_cost_model expects."""
    return pd.DataFrame(
        {
            "Lot": [1, 2, 3, 4, 5, 6],
            "Lot FY": [2015, 2016, 2017, 2018, 2019, 2020],
            "Qty": [5.0, 9.0, 14.0, 22.0, 34.0, 50.0],
            "AUC ($K)": [857.91, 645.57, 531.74, 437.51, 380.10, 332.21],
        }
    )


@pytest.fixture(scope="session")
def estimate_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Lot": [1, 2, 3, 4, 5, 6],
            "Lot FY": [2028, 2029, 2030, 2031, 2032, 2033],
            "Qty": [12.0, 20.0, 30.0, 40.0, 25.0, 10.0],
            "Complexity": [1.15] * 6,
        }
    )
