"""Tier 2 integration tests for the diurnal dynamic simulation module."""

import numpy as np
import pytest

from uct_activated_sludge.diurnal import (
    DiurnalData,
    default_diurnal_data,
    run_diurnal,
)


class TestDefaultDiurnalData:
    """Tests for default_diurnal_data."""

    def test_returns_12_records(self):
        """Should return exactly 12 records for 2-hourly intervals."""
        data = default_diurnal_data()
        assert len(data) == 12

    def test_record_types(self):
        """All records should be DiurnalData instances."""
        data = default_diurnal_data()
        for rec in data:
            assert isinstance(rec, DiurnalData)

    def test_times_are_0_to_22(self):
        """Times should be 0, 2, 4, ..., 22 hours."""
        data = default_diurnal_data()
        expected_times = [float(i * 2) for i in range(12)]
        actual_times = [rec.Time for rec in data]
        assert actual_times == expected_times

    def test_default_values(self):
        """Default flow, COD, TKN should match the function defaults."""
        data = default_diurnal_data()
        for rec in data:
            assert rec.Flow == 15.0
            assert rec.COD == 500.0
            assert rec.TKN == 50.0

    def test_custom_values(self):
        """Custom flow, COD, TKN should be propagated to all records."""
        data = default_diurnal_data(flow=30.0, cod=800.0, tkn=80.0)
        for rec in data:
            assert rec.Flow == 30.0
            assert rec.COD == 800.0
            assert rec.TKN == 80.0


class TestDiurnalData:
    """Tests for the DiurnalData dataclass."""

    def test_default_construction(self):
        """Default DiurnalData should have all zeros."""
        d = DiurnalData()
        assert d.Time == 0.0
        assert d.Flow == 0.0
        assert d.COD == 0.0
        assert d.TKN == 0.0

    def test_custom_construction(self):
        """DiurnalData should accept custom values."""
        d = DiurnalData(Time=6.0, Flow=20.0, COD=600.0, TKN=60.0)
        assert d.Time == 6.0
        assert d.Flow == 20.0
        assert d.COD == 600.0
        assert d.TKN == 60.0


class TestRunDiurnalImport:
    """Basic import tests for the diurnal module."""

    def test_run_diurnal_is_callable(self):
        """run_diurnal should be importable and callable."""
        assert callable(run_diurnal)

    def test_store_response_importable(self):
        """store_response should be importable."""
        from uct_activated_sludge.diurnal import store_response
        assert callable(store_response)

    def test_modify_waste_importable(self):
        """modify_waste should be importable."""
        from uct_activated_sludge.diurnal import modify_waste
        assert callable(modify_waste)
