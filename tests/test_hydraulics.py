"""Tests for the flow division, wastage, and hydraulic calculations."""

import numpy as np
import pytest

from uct_activated_sludge.constants import MAX_REAC_P1
from uct_activated_sludge.hydraulics import flow_division_ss, set_waste


class TestFlowDivisionSS:
    """Tests for flow_division_ss."""

    def _make_arrays(self):
        """Create zero-filled 1-based arrays for flags and feed fractions."""
        frac_feed = np.zeros(MAX_REAC_P1 + 1, dtype=np.float64)
        flag_ras_in = np.zeros(MAX_REAC_P1 + 1, dtype=np.int8)
        flag_a_in = np.zeros(MAX_REAC_P1 + 1, dtype=np.int8)
        flag_b_in = np.zeros(MAX_REAC_P1 + 1, dtype=np.int8)
        flag_a_out = np.zeros(MAX_REAC_P1 + 1, dtype=np.int8)
        flag_b_out = np.zeros(MAX_REAC_P1 + 1, dtype=np.int8)
        return frac_feed, flag_ras_in, flag_a_in, flag_b_in, flag_a_out, flag_b_out

    def test_single_reactor_no_recycles(self):
        """Single reactor with no recycles: all feed flows through."""
        frac_feed, flag_ras_in, flag_a_in, flag_b_in, flag_a_out, flag_b_out = self._make_arrays()
        frac_feed[1] = 1.0

        result = flow_division_ss(
            last_reactor=1,
            flow_feed=100.0,
            flow_ras_recycle=0.0,
            flow_a_recycle=0.0,
            flow_b_recycle=0.0,
            flow_waste=5.0,
            frac_feed=frac_feed,
            flag_ras_in=flag_ras_in,
            flag_a_in=flag_a_in,
            flag_b_in=flag_b_in,
            flag_a_out=flag_a_out,
            flag_b_out=flag_b_out,
        )
        # Flow into reactor 1 = 0 (from previous) + 100 * 1.0 (feed) = 100
        np.testing.assert_allclose(result["FlowInTotal"][1], 100.0)
        # FlowFromPrevious[1] = 0 (first reactor)
        np.testing.assert_allclose(result["FlowFromPrevious"][1], 0.0)

    def test_three_reactor_uct_config(self):
        """3-reactor UCT: feed=25, RAS=25 to R1, A-recycle=75 from R3 to R2."""
        frac_feed, flag_ras_in, flag_a_in, flag_b_in, flag_a_out, flag_b_out = self._make_arrays()
        frac_feed[1] = 1.0
        flag_ras_in[1] = 1
        flag_a_in[2] = 1
        flag_a_out[3] = 1

        result = flow_division_ss(
            last_reactor=3,
            flow_feed=25.0,
            flow_ras_recycle=25.0,
            flow_a_recycle=75.0,
            flow_b_recycle=0.0,
            flow_waste=0.5,
            frac_feed=frac_feed,
            flag_ras_in=flag_ras_in,
            flag_a_in=flag_a_in,
            flag_b_in=flag_b_in,
            flag_a_out=flag_a_out,
            flag_b_out=flag_b_out,
        )
        FIT = result["FlowInTotal"]
        FFP = result["FlowFromPrevious"]

        # Reactor 1: flow_from_prev[1]=0, feed=25, RAS=25 => FIT[1]=50
        np.testing.assert_allclose(FIT[1], 50.0)
        # FlowFromPrevious[2] = FIT[1] - A_out*75 - B_out*0 = 50.0
        np.testing.assert_allclose(FFP[2], 50.0)
        # Reactor 2: FFP[2]=50, A_in=75 => FIT[2]=125
        np.testing.assert_allclose(FIT[2], 125.0)
        # FlowFromPrevious[3] = FIT[2] - 0 = 125
        np.testing.assert_allclose(FFP[3], 125.0)
        # Reactor 3: FIT[3] = FFP[3] + 0 (feed) = 125
        np.testing.assert_allclose(FIT[3], 125.0)
        # FlowFromPrevious[4] (settler) = feed + RAS - waste = 25+25-0.5 = 49.5
        np.testing.assert_allclose(FFP[4], 49.5)

    def test_flow_conservation_three_reactor(self):
        """Total flow in should be traceable through the reactor chain."""
        frac_feed, flag_ras_in, flag_a_in, flag_b_in, flag_a_out, flag_b_out = self._make_arrays()
        frac_feed[1] = 1.0
        flag_ras_in[1] = 1
        flag_a_in[2] = 1
        flag_a_out[3] = 1

        result = flow_division_ss(
            last_reactor=3,
            flow_feed=25.0,
            flow_ras_recycle=25.0,
            flow_a_recycle=75.0,
            flow_b_recycle=0.0,
            flow_waste=0.5,
            frac_feed=frac_feed,
            flag_ras_in=flag_ras_in,
            flag_a_in=flag_a_in,
            flag_b_in=flag_b_in,
            flag_a_out=flag_a_out,
            flag_b_out=flag_b_out,
        )
        FIT = result["FlowInTotal"]
        FFP = result["FlowFromPrevious"]

        # At each reactor: flow_out = FIT[k] - A_out - B_out = FFP[k+1]
        # Reactor 1: out = FIT[1] - 0 = 50 = FFP[2]
        np.testing.assert_allclose(FIT[1], FFP[2])
        # Reactor 2: out = FIT[2] - 0 = 125 = FFP[3]
        np.testing.assert_allclose(FIT[2], FFP[3])
        # Reactor 3: out = FIT[3] - 75 (A-recycle out) = 50 (but FFP[4] is calculated differently)
        # FFP[4] = feed + RAS - waste (settling tank formula)
        np.testing.assert_allclose(FFP[4], 25.0 + 25.0 - 0.5)


class TestSetWaste:
    """Tests for set_waste."""

    def test_all_flows_above_threshold(self):
        """When all dynamic flows exceed waste_avg, wastage is active everywhere."""
        dynamic_flow = np.zeros(13, dtype=np.float64)
        for i in range(1, 13):
            dynamic_flow[i] = 30.0  # all above waste_avg=10

        result = set_waste(
            no_diurnal_ints=12,
            dynamic_flow=dynamic_flow,
            flow_waste_avg=10.0,
        )
        assert result["NoWasteInts"] == 12
        # Total daily wastage distributed evenly
        for i in range(1, 13):
            assert result["WastageOn"][i] is True or result["WastageOn"][i] == True
            np.testing.assert_allclose(result["FlowWaste"][i], 10.0)

    def test_some_flows_below_threshold(self):
        """When some flows are below waste_avg, wastage is only active in high-flow intervals."""
        dynamic_flow = np.zeros(13, dtype=np.float64)
        for i in range(1, 13):
            if i <= 4:
                dynamic_flow[i] = 5.0   # below waste_avg
            else:
                dynamic_flow[i] = 30.0  # above waste_avg

        result = set_waste(
            no_diurnal_ints=12,
            dynamic_flow=dynamic_flow,
            flow_waste_avg=10.0,
        )
        assert result["NoWasteInts"] == 8  # 8 intervals above threshold
        # Wastage in active intervals = avg * 12 / 8
        for i in range(1, 13):
            if i <= 4:
                assert result["WastageOn"][i] == False
                np.testing.assert_allclose(result["FlowWaste"][i], 0.0)
            else:
                assert result["WastageOn"][i] == True
                np.testing.assert_allclose(result["FlowWaste"][i], 10.0 * 12.0 / 8.0)

    def test_distributes_correctly(self):
        """Wastage should distribute average daily mass across active intervals."""
        dynamic_flow = np.zeros(13, dtype=np.float64)
        for i in range(1, 13):
            dynamic_flow[i] = 20.0  # all above threshold
        flow_waste_avg = 5.0

        result = set_waste(
            no_diurnal_ints=12,
            dynamic_flow=dynamic_flow,
            flow_waste_avg=flow_waste_avg,
        )
        # With all 12 intervals active, each gets: 5.0 * 12 / 12 = 5.0
        for i in range(1, 13):
            np.testing.assert_allclose(result["FlowWaste"][i], 5.0)
