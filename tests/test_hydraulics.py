"""Tests for the flow division, wastage, and hydraulic calculations."""

import numpy as np

from uct_activated_sludge.constants import MAX_REAC_P1, NO_DIURNAL_INTS, TOTAL_COMPOUNDS
from uct_activated_sludge.hydraulics import (
    flow_division_dynamic,
    flow_division_ss,
    set_waste,
    wastage_and_flows,
)


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
        dynamic_flow = np.zeros(NO_DIURNAL_INTS + 1, dtype=np.float64)
        for i in range(1, NO_DIURNAL_INTS + 1):
            dynamic_flow[i] = 30.0  # all above waste_avg=10

        result = set_waste(
            no_diurnal_ints=12,
            dynamic_flow=dynamic_flow,
            flow_waste_avg=10.0,
        )
        assert result["NoWasteInts"] == 12
        # Total daily wastage distributed evenly
        for i in range(1, NO_DIURNAL_INTS + 1):
            assert result["WastageOn"][i] is True or result["WastageOn"][i] == True
            np.testing.assert_allclose(result["FlowWaste"][i], 10.0)

    def test_some_flows_below_threshold(self):
        """When some flows are below waste_avg, wastage is only active in high-flow intervals."""
        dynamic_flow = np.zeros(NO_DIURNAL_INTS + 1, dtype=np.float64)
        for i in range(1, NO_DIURNAL_INTS + 1):
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
        for i in range(1, NO_DIURNAL_INTS + 1):
            if i <= 4:
                assert result["WastageOn"][i] == False
                np.testing.assert_allclose(result["FlowWaste"][i], 0.0)
            else:
                assert result["WastageOn"][i] == True
                np.testing.assert_allclose(result["FlowWaste"][i], 10.0 * 12.0 / 8.0)

    def test_distributes_correctly(self):
        """Wastage should distribute average daily mass across active intervals."""
        dynamic_flow = np.zeros(NO_DIURNAL_INTS + 1, dtype=np.float64)
        for i in range(1, NO_DIURNAL_INTS + 1):
            dynamic_flow[i] = 20.0  # all above threshold
        flow_waste_avg = 5.0

        result = set_waste(
            no_diurnal_ints=12,
            dynamic_flow=dynamic_flow,
            flow_waste_avg=flow_waste_avg,
        )
        # With all 12 intervals active, each gets: 5.0 * 12 / 12 = 5.0
        for i in range(1, NO_DIURNAL_INTS + 1):
            np.testing.assert_allclose(result["FlowWaste"][i], 5.0)


class TestFlowDivisionDynamic:
    """Tests for flow_division_dynamic."""

    def _make_arrays(self):
        """Create zero-filled 1-based arrays for flags and feed fractions."""
        frac_feed = np.zeros(MAX_REAC_P1 + 1, dtype=np.float64)
        vol = np.zeros(MAX_REAC_P1 + 1, dtype=np.float64)
        flag_ras_in = np.zeros(MAX_REAC_P1 + 1, dtype=np.int8)
        flag_a_in = np.zeros(MAX_REAC_P1 + 1, dtype=np.int8)
        flag_b_in = np.zeros(MAX_REAC_P1 + 1, dtype=np.int8)
        flag_a_out = np.zeros(MAX_REAC_P1 + 1, dtype=np.int8)
        flag_b_out = np.zeros(MAX_REAC_P1 + 1, dtype=np.int8)
        return frac_feed, vol, flag_ras_in, flag_a_in, flag_b_in, flag_a_out, flag_b_out

    def test_three_reactor_uct_dilution_rates(self):
        """3-reactor UCT: verify dilution rates are computed correctly."""
        frac_feed, vol, flag_ras_in, flag_a_in, flag_b_in, flag_a_out, flag_b_out = (
            self._make_arrays()
        )
        frac_feed[1] = 1.0
        vol[1] = 1.5
        vol[2] = 3.0
        vol[3] = 6.0
        flag_ras_in[1] = 1
        flag_a_in[2] = 1
        flag_a_out[3] = 1

        result = flow_division_dynamic(
            last_reactor=3,
            flow_feed=25.0,
            flow_ras_recycle=25.0,
            flow_a_recycle=75.0,
            flow_b_recycle=0.0,
            wastage_rate=0.5,
            frac_feed=frac_feed,
            vol=vol,
            flag_ras_in=flag_ras_in,
            flag_a_in=flag_a_in,
            flag_b_in=flag_b_in,
            flag_a_out=flag_a_out,
            flag_b_out=flag_b_out,
        )

        # Verify expected keys
        expected_keys = {
            "FlowInTotal", "FlowFromPrevious",
            "DhOut", "DhFromPrevious", "DhFeed",
            "DhRAS", "DhArecycle", "DhBrecycle",
        }
        assert expected_keys == set(result.keys())

        # Reactor 1: FIT = 0 + 25 (RAS) + 25 (feed) = 50
        # DhOut[1] = 50 / 1.5 = 33.333...
        np.testing.assert_allclose(result["FlowInTotal"][1], 50.0)
        np.testing.assert_allclose(result["DhOut"][1], 50.0 / 1.5, rtol=1e-12)

        # DhFeed[1] = 25 * 1.0 / 1.5 = 16.666...
        np.testing.assert_allclose(result["DhFeed"][1], 25.0 / 1.5, rtol=1e-12)

        # DhFromPrevious[1] = 0 / 1.5 = 0
        np.testing.assert_allclose(result["DhFromPrevious"][1], 0.0)

        # Reactor 2: FIT = 50 + 75 (A-recycle) = 125
        # DhOut[2] = 125 / 3.0 = 41.666...
        np.testing.assert_allclose(result["FlowInTotal"][2], 125.0)
        np.testing.assert_allclose(result["DhOut"][2], 125.0 / 3.0, rtol=1e-12)

        # DhArecycle[2] = 75 * 1 / 3.0 = 25.0
        np.testing.assert_allclose(result["DhArecycle"][2], 25.0, rtol=1e-12)

        # Reactor 3: FIT = 125 + 0 = 125
        # DhOut[3] = 125 / 6.0 = 20.833...
        np.testing.assert_allclose(result["FlowInTotal"][3], 125.0)
        np.testing.assert_allclose(result["DhOut"][3], 125.0 / 6.0, rtol=1e-12)

    def test_dh_ras_particulate_vs_soluble(self):
        """DhRAS should differ for particulate (1-7) vs soluble (8-13) compounds."""
        frac_feed, vol, flag_ras_in, flag_a_in, flag_b_in, flag_a_out, flag_b_out = (
            self._make_arrays()
        )
        frac_feed[1] = 1.0
        vol[1] = 1.5
        vol[2] = 3.0
        vol[3] = 6.0
        flag_ras_in[1] = 1
        flag_a_in[2] = 1
        flag_a_out[3] = 1

        result = flow_division_dynamic(
            last_reactor=3,
            flow_feed=25.0,
            flow_ras_recycle=25.0,
            flow_a_recycle=75.0,
            flow_b_recycle=0.0,
            wastage_rate=0.5,
            frac_feed=frac_feed,
            vol=vol,
            flag_ras_in=flag_ras_in,
            flag_a_in=flag_a_in,
            flag_b_in=flag_b_in,
            flag_a_out=flag_a_out,
            flag_b_out=flag_b_out,
        )

        DhRAS = result["DhRAS"]

        # For reactor 1 (which has RAS in):
        # Particulate: (25 + 25 - 0.5) * 1 / 1.5 = 49.5 / 1.5 = 33.0
        # Soluble: 25 * 1 / 1.5 = 16.666...
        from uct_activated_sludge.constants import NO_PART

        # Particulate compounds (1..7)
        for i in range(1, NO_PART + 1):
            np.testing.assert_allclose(DhRAS[1, i], 49.5 / 1.5, rtol=1e-12)

        # Soluble compounds (8..13)
        for i in range(NO_PART + 1, 14):
            np.testing.assert_allclose(DhRAS[1, i], 25.0 / 1.5, rtol=1e-12)

        # Reactor 2 has no RAS in, so DhRAS should be 0
        for i in range(1, 14):
            np.testing.assert_allclose(DhRAS[2, i], 0.0)


class TestWastageAndFlows:
    """Tests for wastage_and_flows."""

    def test_wastage_converges(self):
        """wastage_and_flows should converge and return a positive waste flow."""
        from uct_activated_sludge.models import (
            KineticParams,
            PlantConfig,
            StoichiometricParams,
            _zeros_2d_1based,
        )
        from uct_activated_sludge.stoichiometry import build_stoichiometric_matrix
        from uct_activated_sludge.kinetics import air_supply
        from uct_activated_sludge.solver import newton as newton_fn, scale_values

        pc = PlantConfig()
        pc.LastReactor = 3
        pc.Vol[1] = 1.5
        pc.Vol[2] = 3.0
        pc.Vol[3] = 6.0
        pc.FracFeed[1] = 1.0
        pc.DOConc[1] = 0.0
        pc.DOConc[2] = 0.0
        pc.DOConc[3] = 2.0
        pc.ReactorAerated[1] = False
        pc.ReactorAerated[2] = False
        pc.ReactorAerated[3] = True
        pc.FlowFeed = 25.0
        pc.FlowRASrecycle = 25.0
        pc.FlowArecycle = 75.0
        pc.FlowBrecycle = 0.0
        pc.FlagRASIn[1] = 1
        pc.FlagAIn[2] = 1
        pc.FlagAOut[3] = 1
        pc.ReactorAIn = 2
        pc.ReactorAOut = 3
        pc.Rs = 20.0
        pc.Temp = 20.0
        pc.VolumeTotal = 10.5

        kp = KineticParams()
        sp = StoichiometricParams()
        stoich = build_stoichiometric_matrix(sp)

        air_on_h, air_off_h, air_on_a, air_off_a = air_supply(pc, kp)

        C = _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS)
        C0 = np.zeros(TOTAL_COMPOUNDS + 1, dtype=np.float64)

        def _newton(compounds):
            newton_fn(
                C, compounds, pc, kp, sp, stoich, C0,
                air_on_h, air_off_h, air_on_a, air_off_a,
            )

        def _scale(compounds):
            scale_values(C, pc, compounds)

        flow_waste = wastage_and_flows(
            plant=pc,
            C=C,
            newton_fn=_newton,
            scale_values_fn=_scale,
        )

        # Should return a positive waste flow
        assert flow_waste > 0.0
        # Should be approximately VolumeTotal / Rs for this config
        np.testing.assert_allclose(flow_waste, pc.VolumeTotal / pc.Rs, rtol=0.1)
        # FlowWasteAvg should be set
        assert pc.FlowWasteAvg > 0.0
