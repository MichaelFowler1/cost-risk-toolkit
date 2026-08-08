"""Loading and persisting cost data.

The schema check is the guard between a spreadsheet someone typed by hand and
a forecast presented as fact, so the tests here care most about what gets
*rejected*.
"""
import pandas as pd
import pytest

from cost_core.data_io import (ValidationError, load_cost_csv,
                               load_from_sqlite, save_to_sqlite)

GOOD = {
    "program": ["Alpha", "Alpha", "Alpha"],
    "lot": [1, 2, 3],
    "unit_quantity": [10.0, 20.0, 40.0],
    "unit_cost": [1000.0, 850.0, 722.5],
}


def write_csv(tmp_path, data, name="costs.csv"):
    path = tmp_path / name
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def test_a_valid_file_loads_with_its_required_columns(tmp_path):
    df = load_cost_csv(write_csv(tmp_path, GOOD))
    assert len(df) == 3
    for col in ("program", "lot", "unit_quantity", "unit_cost"):
        assert col in df.columns


def test_a_missing_required_column_is_named_in_the_error(tmp_path):
    data = {k: v for k, v in GOOD.items() if k != "unit_cost"}
    with pytest.raises(ValidationError, match="unit_cost"):
        load_cost_csv(write_csv(tmp_path, data))


def test_negative_costs_are_rejected(tmp_path):
    bad = dict(GOOD, unit_cost=[1000.0, -850.0, 722.5])
    with pytest.raises(ValidationError, match="Negative"):
        load_cost_csv(write_csv(tmp_path, bad))


def test_negative_quantities_are_rejected(tmp_path):
    bad = dict(GOOD, unit_quantity=[10.0, -20.0, 40.0])
    with pytest.raises(ValidationError, match="Negative"):
        load_cost_csv(write_csv(tmp_path, bad))


def test_extra_columns_are_dropped_rather_than_carried(tmp_path):
    """Unknown columns shouldn't silently ride along into a model fit."""
    noisy = dict(GOOD, analyst_scratch=["x", "y", "z"])
    df = load_cost_csv(write_csv(tmp_path, noisy))
    assert "analyst_scratch" not in df.columns


def test_optional_columns_are_kept(tmp_path):
    withopt = dict(GOOD, notes=["a", "b", "c"])
    df = load_cost_csv(write_csv(tmp_path, withopt))
    assert "notes" in df.columns


def test_sqlite_round_trip_preserves_the_numbers(tmp_path):
    df = load_cost_csv(write_csv(tmp_path, GOOD))
    db = tmp_path / "costs.sqlite"
    save_to_sqlite(df, db)
    back = load_from_sqlite(db)
    assert len(back) == len(df)
    assert back["unit_cost"].tolist() == pytest.approx(df["unit_cost"].tolist())
    assert back["unit_quantity"].tolist() == pytest.approx(
        df["unit_quantity"].tolist())
