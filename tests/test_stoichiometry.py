"""Tests for the ASM1 stoichiometric matrix construction."""

import numpy as np
import numpy.typing as npt
import pytest

from uct_activated_sludge.constants import NO_PROCESSES, TOTAL_COMPOUNDS
from uct_activated_sludge.models import StoichiometricParams
from uct_activated_sludge.stoichiometry import build_stoichiometric_matrix


class TestBuildStoichiometricMatrix:
    """Tests for build_stoichiometric_matrix."""

    def test_matrix_shape(self, default_stoich_params):
        """Matrix shape should be (TOTAL_COMPOUNDS+1, NO_PROCESSES+1) for 1-based indexing."""
        Stoich = build_stoichiometric_matrix(default_stoich_params)
        assert Stoich.shape == (TOTAL_COMPOUNDS + 1, NO_PROCESSES + 1)
        assert Stoich.shape == (15, 15)

    def test_stoich_8_1_is_neg_inv_yh(self, default_stoich_params, default_stoich_matrix):
        """Stoich[8,1] should be -1/Yh (Ss consumed in aerobic growth on Ss)."""
        Yh = default_stoich_params.Yh
        expected = -1.0 / Yh
        np.testing.assert_allclose(default_stoich_matrix[8, 1], expected, rtol=1e-12)

    def test_stoich_1_1_is_one(self, default_stoich_matrix):
        """Stoich[1,1] should be 1.0 (Xbh produced in aerobic growth on Ss)."""
        np.testing.assert_allclose(default_stoich_matrix[1, 1], 1.0, atol=1e-15)

    def test_stoich_14_13_autotrophic_do(self, default_stoich_params, default_stoich_matrix):
        """Stoich[14,13] should be -(4.57-Ya)/Ya (DO consumed in autotrophic growth)."""
        Ya = default_stoich_params.Ya
        expected = -(4.57 - Ya) / Ya
        np.testing.assert_allclose(default_stoich_matrix[14, 13], expected, rtol=1e-12)

    def test_nonzero_entry_count(self, default_stoich_matrix):
        """The stoichiometric matrix should have a specific number of non-zero entries.

        Count only the 1-based portion Stoich[1:, 1:].
        """
        active = default_stoich_matrix[1:, 1:]
        nonzero_count = np.count_nonzero(active)
        # From inspection of the source: each process has several non-zero entries.
        # Processes 1-8: 4-5 entries each, Process 9: 4, Process 10: 2,
        # Process 11: 2, Process 12: 3, Process 13: 4, Process 14: 4
        assert nonzero_count > 40  # there are many non-zero stoichiometric coefficients
        assert nonzero_count < 100  # but the matrix is sparse

    def test_row_0_and_col_0_are_zero(self, default_stoich_matrix):
        """Row 0 and column 0 should be all zeros (unused padding)."""
        np.testing.assert_array_equal(default_stoich_matrix[0, :], 0.0)
        np.testing.assert_array_equal(default_stoich_matrix[:, 0], 0.0)

    def test_decay_process_conserves_biomass(self, default_stoich_params, default_stoich_matrix):
        """Heterotrophic decay (process 9) should have COD balance:
        -1 (Xbh) + Fe (Xe) + (1-Fe) (Sbp) = 0 in COD terms.
        """
        Fe = default_stoich_params.Fe
        S = default_stoich_matrix
        # Stoich[1,9] = -1.0 (Xbh consumed)
        # Stoich[2,9] = Fe   (Xe produced)
        # Stoich[5,9] = 1-Fe (Sbp produced)
        cod_balance = S[1, 9] + S[2, 9] + S[5, 9]
        np.testing.assert_allclose(cod_balance, 0.0, atol=1e-14)

    def test_autotrophic_decay_conserves_biomass(self, default_stoich_params, default_stoich_matrix):
        """Autotrophic decay (process 14) should conserve COD in biomass terms."""
        Fe = default_stoich_params.Fe
        S = default_stoich_matrix
        # Stoich[3,14] = -1.0 (Xba consumed)
        # Stoich[2,14] = Fe   (Xe produced)
        # Stoich[5,14] = 1-Fe (Sbp produced)
        cod_balance = S[3, 14] + S[2, 14] + S[5, 14]
        np.testing.assert_allclose(cod_balance, 0.0, atol=1e-14)

    def test_process_1_entries(self, default_stoich_params, default_stoich_matrix):
        """Process 1: Aerobic growth of heterotrophs on Ss with NH3."""
        Yh = default_stoich_params.Yh
        Ixb = default_stoich_params.Ixb
        S = default_stoich_matrix

        np.testing.assert_allclose(S[1, 1], 1.0)
        np.testing.assert_allclose(S[8, 1], -1.0 / Yh)
        np.testing.assert_allclose(S[9, 1], -Ixb)
        np.testing.assert_allclose(S[12, 1], -Ixb / 14.0)
        np.testing.assert_allclose(S[14, 1], -(1.0 - Yh) / Yh)

    def test_storage_process_entries(self, default_stoich_matrix):
        """Process 10: Storage of particulate COD. Xs gained, Sbp consumed."""
        S = default_stoich_matrix
        np.testing.assert_allclose(S[4, 10], 1.0)
        np.testing.assert_allclose(S[5, 10], -1.0)

    def test_hydrolysis_process_entries(self, default_stoich_matrix):
        """Process 11: Hydrolysis of particulate organic N."""
        S = default_stoich_matrix
        np.testing.assert_allclose(S[7, 11], -1.0)
        np.testing.assert_allclose(S[10, 11], 1.0)

    def test_ammonification_process_entries(self, default_stoich_matrix):
        """Process 12: Ammonification of soluble organic N."""
        S = default_stoich_matrix
        np.testing.assert_allclose(S[9, 12], 1.0)
        np.testing.assert_allclose(S[10, 12], -1.0)
        np.testing.assert_allclose(S[12, 12], 1.0 / 14.0)

    def test_custom_params(self):
        """Matrix entries should reflect custom stoichiometric parameters."""
        params = StoichiometricParams(Yh=0.5, Fe=0.1, Ya=0.2)
        S = build_stoichiometric_matrix(params)
        np.testing.assert_allclose(S[8, 1], -1.0 / 0.5)  # -1/Yh
        np.testing.assert_allclose(S[2, 9], 0.1)          # Fe for decay
        np.testing.assert_allclose(S[14, 13], -(4.57 - 0.2) / 0.2)
