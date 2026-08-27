"""ASM1 stoichiometric matrix construction.

Ported from the ``Stoichiometry`` procedure in STCHUNIT.PAS.

The matrix ``Stoich[compound, process]`` has shape
(TOTAL_COMPOUNDS+1, NO_PROCESSES+1) so that 1-based Pascal indices
can be used directly (row/column 0 is unused padding).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from uct_activated_sludge.constants import NO_PROCESSES, TOTAL_COMPOUNDS
from uct_activated_sludge.models import StoichiometricParams

__all__ = ["build_stoichiometric_matrix"]


def build_stoichiometric_matrix(
    params: StoichiometricParams,
) -> npt.NDArray[np.float64]:
    """Build the 14x14 ASM1 stoichiometric matrix.

    Parameters
    ----------
    params : StoichiometricParams
        Yield and stoichiometric coefficients (Yh, Fe, Ixb, Ixe, Ya, ...).

    Returns
    -------
    numpy.ndarray
        2-D float64 array of shape ``(TOTAL_COMPOUNDS+1, NO_PROCESSES+1)``.
        Index 0 in both dimensions is unused padding.  Valid entries are
        ``Stoich[1..14, 1..14]``.
    """
    Stoich = np.zeros(
        (TOTAL_COMPOUNDS + 1, NO_PROCESSES + 1), dtype=np.float64
    )

    # Unpack parameters for readability (matches Pascal local names).
    Yh = params.Yh
    Fe = params.Fe
    Ixb = params.Ixb
    Ixe = params.Ixe
    Ya = params.Ya

    # ------------------------------------------------------------------
    # Process 1: Aerobic growth of heterotrophs on Ss with NH3
    # ------------------------------------------------------------------
    Stoich[1, 1] = 1.0
    Stoich[8, 1] = -1.0 / Yh
    Stoich[9, 1] = -Ixb
    Stoich[12, 1] = -Ixb / 14.0
    Stoich[14, 1] = -(1.0 - Yh) / Yh

    # ------------------------------------------------------------------
    # Process 2: Aerobic growth of heterotrophs on Ss with NO3
    # ------------------------------------------------------------------
    Stoich[1, 2] = 1.0
    Stoich[8, 2] = -1.0 / Yh
    Stoich[11, 2] = -Ixb
    Stoich[12, 2] = +Ixb / 14.0
    Stoich[14, 2] = -(1.0 - Yh) / Yh

    # ------------------------------------------------------------------
    # Process 3: Anoxic growth of heterotrophs on Ss with NH3
    # ------------------------------------------------------------------
    Stoich[1, 3] = 1.0
    Stoich[8, 3] = -1.0 / Yh
    Stoich[9, 3] = -Ixb
    Stoich[11, 3] = -(1 - Yh) / (2.86 * Yh)
    Stoich[12, 3] = (1.0 - Yh) / (14.0 * 2.86 * Yh) - Ixb / 14.0

    # ------------------------------------------------------------------
    # Process 4: Anoxic growth of heterotrophs on Ss with NO3
    # ------------------------------------------------------------------
    Stoich[1, 4] = 1.0
    Stoich[8, 4] = -1.0 / Yh
    Stoich[11, 4] = -(1 - Yh) / (2.86 * Yh) - Ixb
    Stoich[12, 4] = (1.0 - Yh) / (14.0 * 2.86 * Yh) + Ixb / 14.0

    # ------------------------------------------------------------------
    # Process 5: Aerobic growth of heterotrophs on Xs with NH3
    # ------------------------------------------------------------------
    Stoich[1, 5] = 1.0
    Stoich[4, 5] = -1.0 / Yh
    Stoich[9, 5] = -Ixb
    Stoich[12, 5] = -Ixb / 14.0
    Stoich[14, 5] = -(1.0 - Yh) / Yh

    # ------------------------------------------------------------------
    # Process 6: Aerobic growth of heterotrophs on Xs with NO3
    # ------------------------------------------------------------------
    Stoich[1, 6] = 1.0
    Stoich[4, 6] = -1.0 / Yh
    Stoich[11, 6] = -Ixb
    Stoich[12, 6] = +Ixb / 14.0
    Stoich[14, 6] = -(1.0 - Yh) / Yh

    # ------------------------------------------------------------------
    # Process 7: Anoxic growth of heterotrophs on Xs with NH3
    # ------------------------------------------------------------------
    Stoich[1, 7] = 1.0
    Stoich[4, 7] = -1.0 / Yh
    Stoich[9, 7] = -Ixb
    Stoich[11, 7] = -(1 - Yh) / (2.86 * Yh)
    Stoich[12, 7] = (1.0 - Yh) / (14.0 * 2.86 * Yh) - Ixb / 14.0

    # ------------------------------------------------------------------
    # Process 8: Anoxic growth of heterotrophs on Xs with NO3
    # ------------------------------------------------------------------
    Stoich[1, 8] = 1.0
    Stoich[4, 8] = -1.0 / Yh
    Stoich[11, 8] = -(1 - Yh) / (2.86 * Yh) - Ixb
    Stoich[12, 8] = (1.0 - Yh) / (14.0 * 2.86 * Yh) + Ixb / 14.0

    # ------------------------------------------------------------------
    # Process 9: Decay of heterotrophs
    # ------------------------------------------------------------------
    Stoich[1, 9] = -1.0
    Stoich[2, 9] = Fe
    Stoich[5, 9] = 1.0 - Fe
    Stoich[7, 9] = Ixb - Fe * Ixe

    # ------------------------------------------------------------------
    # Process 10: Storage of particulate influent COD
    # ------------------------------------------------------------------
    Stoich[4, 10] = 1.0
    Stoich[5, 10] = -1.0

    # ------------------------------------------------------------------
    # Process 11: Hydrolysis of biodegradable particulate organic N
    # ------------------------------------------------------------------
    Stoich[7, 11] = -1.0
    Stoich[10, 11] = 1.0

    # ------------------------------------------------------------------
    # Process 12: Ammonification of soluble organic N
    # ------------------------------------------------------------------
    Stoich[9, 12] = 1.0
    Stoich[10, 12] = -1.0
    Stoich[12, 12] = 1.0 / 14.0

    # ------------------------------------------------------------------
    # Process 13: Autotrophic growth
    # ------------------------------------------------------------------
    Stoich[3, 13] = 1.0
    Stoich[9, 13] = -(Ixb + 1.0 / Ya)
    Stoich[11, 13] = 1.0 / Ya
    Stoich[12, 13] = -(Ixb / 14.0 + 1.0 / (7.0 * Ya))
    Stoich[14, 13] = -(4.57 - Ya) / Ya

    # ------------------------------------------------------------------
    # Process 14: Autotrophic decay
    # ------------------------------------------------------------------
    Stoich[2, 14] = Fe
    Stoich[3, 14] = -1.0
    Stoich[5, 14] = 1.0 - Fe
    Stoich[7, 14] = Ixb - Fe * Ixe

    return Stoich
