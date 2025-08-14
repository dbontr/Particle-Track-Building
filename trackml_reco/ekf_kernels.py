from __future__ import annotations
import numpy as np
from typing import Tuple

# Optional Numba acceleration (safe fallback to NumPy)
try:
    from numba import njit, prange  # type: ignore
    NUMBA_OK = True
except Exception:  # pragma: no cover
    NUMBA_OK = False


__all__ = [
    "chi2_batch",
    "kalman_gain",
    "factor_S",
    "chi2_batch_factored",
    "kalman_gain_factored",
]


def _robust_cholesky_numpy(S: np.ndarray) -> np.ndarray:
    r"""
    Robust Cholesky factorization with small diagonal *jitter* and SPD fallback.

    Attempts ``np.linalg.cholesky(S)``; on failure, retries with
    :math:`S+\varepsilon I_3` where :math:`\varepsilon` is escalated
    geometrically. If all retries fail, an eigenvalue floor is applied:

    .. math::
        S_\text{fix} = V\;\mathrm{diag}(\max(w,\; w_\max\,10^{-15}))\;V^\top,

    where :math:`V` and :math:`w` are eigenvectors/values of :math:`S`.

    Parameters
    ----------
    S : ndarray, shape (3, 3)
        Symmetric innovation covariance (not necessarily strictly SPD).

    Returns
    -------
    L : ndarray, shape (3, 3)
        Lower-triangular Cholesky factor satisfying
        :math:`S_\text{spd}=L L^\top` with :math:`S_\text{spd}\approx S`.

    Notes
    -----
    The returned :math:`L` can be used to perform triangular solves instead of
    inverting :math:`S` explicitly (more stable and faster for small systems).
    """
    try:
        return np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # escalate jitter geometrically; keeps units of S
        I = np.eye(3, dtype=S.dtype)
        eps = 1e-12
        for _ in range(8):
            try:
                return np.linalg.cholesky(S + eps * I)
            except np.linalg.LinAlgError:
                eps *= 10.0
        # last resort: use eigen floor
        w, V = np.linalg.eigh(S)
        w = np.clip(w, w.max() * 1e-15, None)
        S_fix = (V * w) @ V.T
        return np.linalg.cholesky(S_fix)


def factor_S(S: np.ndarray) -> np.ndarray:
    r"""
    Factor the innovation covariance :math:`S` into its Cholesky factor.

    Parameters
    ----------
    S : array_like, shape (3, 3)
        Innovation covariance :math:`S = H P^- H^\top + R`.

    Returns
    -------
    L : ndarray, shape (3, 3)
        Lower-triangular matrix :math:`L` such that :math:`S \approx L L^\top`
        (robust to small non-SPD perturbations via jitter/eigen floor).

    See Also
    --------
    chi2_batch_factored : Batched Mahalanobis using a precomputed :math:`L`.
    kalman_gain_factored : EKF gain using a precomputed :math:`L`.

    Notes
    -----
    Using :math:`L` enables *two triangular solves* in lieu of an explicit
    inverse, e.g. solving :math:`S\,x=b` via :math:`L(L^\top x)=b`.
    """
    S = np.asarray(S, dtype=np.float64, order="C")
    return _robust_cholesky_numpy(S)


def _chi2_batch_numpy(diff: np.ndarray, S: np.ndarray) -> np.ndarray:
    r"""
    Batched Mahalanobis :math:`\chi^2` with robust Cholesky and triangular solves.

    Given residuals :math:`r_i \in \mathbb{R}^3` (rows of ``diff``) and
    covariance :math:`S\in\mathbb{R}^{3\times 3}`, the Mahalanobis distance is

    .. math::
        \chi^2_i \;=\; r_i^\top S^{-1} r_i \;=\; \|L^{-1} r_i\|_2^2,
        \qquad S = L L^\top.

    This implementation computes all :math:`\chi^2_i` via two triangular solves
    using a *robust* Cholesky factor :math:`L`.

    Parameters
    ----------
    diff : ndarray, shape (m, 3)
        Residuals stacked row-wise.
    S : ndarray, shape (3, 3)
        Innovation/measurement covariance.

    Returns
    -------
    chi2 : ndarray, shape (m,)
        Mahalanobis :math:`\chi^2` values for each row in ``diff``.

    Notes
    -----
    The computation proceeds as: ``Y = solve(L, diff.T)``, ``X = solve(L.T, Y)``,
    and :math:`\chi^2_i = r_i^\top x_i = \sum_j y_{ji}^2`. Using the
    Cholesky-based solves avoids forming :math:`S^{-1}` explicitly.
    """
    if diff.size == 0:
        return np.empty(0, dtype=np.float64)
    diff = np.asarray(diff, dtype=np.float64, order="C")
    L = _robust_cholesky_numpy(np.asarray(S, dtype=np.float64, order="C"))
    # Solve L * Y = diff^T  -> Y = L^{-1} diff^T
    # Solve L^T * X = Y     -> X = S^{-1} diff^T
    Y = np.linalg.solve(L, diff.T)          # (3, m)
    X = np.linalg.solve(L.T, Y)             # (3, m)
    # chi2_i = diff_i^T * x_i
    return np.einsum("ij,ij->i", diff, X.T, optimize=True)


def _chi2_batch_factored_numpy(diff: np.ndarray, L: np.ndarray) -> np.ndarray:
    r"""
    Batched Mahalanobis :math:`\chi^2` using a **precomputed** Cholesky factor.

    Parameters
    ----------
    diff : ndarray, shape (m, 3)
        Residuals stacked row-wise.
    L : ndarray, shape (3, 3)
        Lower-triangular factor with :math:`S = L L^\top`.

    Returns
    -------
    chi2 : ndarray, shape (m,)
        Mahalanobis :math:`\chi^2` values.

    Notes
    -----
    Avoids factoring :math:`S` repeatedly when evaluating multiple residual
    sets against the *same* covariance.
    """
    if diff.size == 0:
        return np.empty(0, dtype=np.float64)
    diff = np.asarray(diff, dtype=np.float64, order="C")
    L = np.asarray(L, dtype=np.float64, order="C")
    Y = np.linalg.solve(L, diff.T)
    X = np.linalg.solve(L.T, Y)
    return np.einsum("ij,ij->i", diff, X.T, optimize=True)


def _kalman_gain_numpy(P_pred: np.ndarray, H: np.ndarray, S: np.ndarray) -> np.ndarray:
    r"""
    EKF gain via Cholesky solves: :math:`K = P^- H^\top S^{-1}`.

    Computes :math:`K\in\mathbb{R}^{7\times 3}` without forming :math:`S^{-1}`
    explicitly by solving :math:`S X^\top = (P^- H^\top)^\top` using the
    robust Cholesky factor :math:`S=L L^\top`.

    Parameters
    ----------
    P_pred : ndarray, shape (7, 7)
        Predicted covariance :math:`P^-`.
    H : ndarray, shape (3, 7)
        Measurement Jacobian (position extractor).
    S : ndarray, shape (3, 3)
        Innovation covariance :math:`S = H P^- H^\top + R`.

    Returns
    -------
    K : ndarray, shape (7, 3)
        Kalman gain.

    Notes
    -----
    Procedure:

    1. Compute :math:`P^- H^\top\in\mathbb{R}^{7\times 3}`.
    2. Solve :math:`L\,Y = (P^- H^\top)^\top` and :math:`L^\top X = Y`.
    3. Return :math:`K = X^\top`.

    This avoids instability from inverting :math:`S` and exploits its small size.
    """
    P_pred = np.asarray(P_pred, dtype=np.float64, order="C")
    H = np.asarray(H, dtype=np.float64, order="C")
    S = np.asarray(S, dtype=np.float64, order="C")

    L = _robust_cholesky_numpy(S)
    PHt = P_pred @ H.T                         # (7,3)
    # Solve S * X^T = (PHt)^T  for X (7,3): two triangular solves on matrix RHS
    Y = np.linalg.solve(L, PHt.T)              # (3,7)
    X = np.linalg.solve(L.T, Y)                # (3,7)
    return X.T                                 # (7,3)


def _kalman_gain_factored_numpy(P_pred: np.ndarray, H: np.ndarray, L: np.ndarray) -> np.ndarray:
    r"""
    EKF gain using a **precomputed** Cholesky factor :math:`L` of :math:`S`.

    Parameters
    ----------
    P_pred : ndarray, shape (7, 7)
        Predicted covariance :math:`P^-`.
    H : ndarray, shape (3, 7)
        Measurement Jacobian.
    L : ndarray, shape (3, 3)
        Lower-triangular factor s.t. :math:`S=L L^\top`.

    Returns
    -------
    K : ndarray, shape (7, 3)
        Kalman gain :math:`K = P^- H^\top S^{-1}` computed via two triangular solves.

    See Also
    --------
    factor_S : Obtain :math:`L` once when reusing the same :math:`S`.
    """
    P_pred = np.asarray(P_pred, dtype=np.float64, order="C")
    H = np.asarray(H, dtype=np.float64, order="C")
    L = np.asarray(L, dtype=np.float64, order="C")

    PHt = P_pred @ H.T                         # (7,3)
    Y = np.linalg.solve(L, PHt.T)              # (3,7)
    X = np.linalg.solve(L.T, Y)                # (3,7)
    return X.T


if NUMBA_OK:
    @njit(cache=True, fastmath=True, parallel=True)
    def _robust_cholesky_numba(S: np.ndarray) -> np.ndarray:
        r"""
        Numba-compatible robust Cholesky with jitter escalation and SPD fallback.

        Attempts ``np.linalg.cholesky(S)`` and, on failure, tries
        :math:`S+\varepsilon I_3` with geometric :math:`\varepsilon` escalation.
        If that fails, clamps eigenvalues to :math:`\max(w,\; w_\max\,10^{-15})`.

        Parameters
        ----------
        S : ndarray, shape (3, 3)

        Returns
        -------
        L : ndarray, shape (3, 3)
            Lower-triangular Cholesky factor.
        """
        # try no-jitter path first
        try:
            return np.linalg.cholesky(S)
        except Exception:
            pass

        I = np.eye(3, dtype=np.float64)
        eps = 1e-12
        for _ in range(8):
            try:
                return np.linalg.cholesky(S + eps * I)
            except Exception:
                eps *= 10.0
        # crude SPD floor via eigen clamp (Numba supports eigh)
        w, V = np.linalg.eigh(S)
        w = np.maximum(w, np.max(w) * 1e-15)
        S_fix = (V * w) @ V.T
        return np.linalg.cholesky(S_fix)

    @njit(cache=True, fastmath=True, parallel=True)
    def chi2_batch(diff: np.ndarray, S: np.ndarray) -> np.ndarray:
        r"""
        Parallel batched Mahalanobis :math:`\chi^2` via Cholesky with manual solves.

        For residuals :math:`r_i\in\mathbb{R}^3` and :math:`S=L L^\top`:

        .. math::
            \chi^2_i \;=\; r_i^\top S^{-1} r_i
            \;=\; \|L^{-1} r_i\|_2^2.

        This version unrolls forward/back substitution for :math:`3\times 3`
        triangular systems and parallelizes over rows with ``prange``.

        Parameters
        ----------
        diff : ndarray, shape (m, 3)
            Residuals stacked row-wise.
        S : ndarray, shape (3, 3)
            Covariance to reuse across the batch.

        Returns
        -------
        chi2 : ndarray, shape (m,)
            Mahalanobis :math:`\chi^2` for each residual.
        """
        m = diff.shape[0]
        out = np.empty(m, dtype=np.float64)
        if m == 0:
            return out
        L = _robust_cholesky_numba(S)

        # Manual 3x3 forward/back substitution (fully unrolled) per diff[i]
        for i in prange(m):
            y0 = diff[i, 0]
            y1 = diff[i, 1]
            y2 = diff[i, 2]

            # forward: L * u = y
            u0 = y0 / L[0, 0]
            u1 = (y1 - L[1, 0] * u0) / L[1, 1]
            u2 = (y2 - L[2, 0] * u0 - L[2, 1] * u1) / L[2, 2]

            # back: L^T * x = u
            x2 = u2 / L[2, 2]
            x1 = (u1 - L[2, 1] * x2) / L[1, 1]
            x0 = (u0 - L[1, 0] * x1 - L[2, 0] * x2) / L[0, 0]

            out[i] = y0 * x0 + y1 * x1 + y2 * x2
        return out

    @njit(cache=True, fastmath=True)
    def kalman_gain(P_pred: np.ndarray, H: np.ndarray, S: np.ndarray) -> np.ndarray:
        r"""
        EKF gain :math:`K = P^- H^\top S^{-1}` via Cholesky solves (Numba).

        Solves :math:`S\,x=b` with forward/back substitution using
        :math:`S=L L^\top`, for each row of :math:`b=(P^- H^\top)\in\mathbb{R}^{7\times 3}`.

        Parameters
        ----------
        P_pred : ndarray, shape (7, 7)
            Predicted covariance :math:`P^-`.
        H : ndarray, shape (3, 7)
            Measurement Jacobian.
        S : ndarray, shape (3, 3)
            Innovation covariance.

        Returns
        -------
        K : ndarray, shape (7, 3)
            Kalman gain.
        """
        L = _robust_cholesky_numba(S)
        PHt = P_pred @ H.T  # (7,3)
        K = np.empty((7, 3), dtype=np.float64)
        for r in range(7):
            b0 = PHt[r, 0]
            b1 = PHt[r, 1]
            b2 = PHt[r, 2]

            # forward: L * u = b
            u0 = b0 / L[0, 0]
            u1 = (b1 - L[1, 0] * u0) / L[1, 1]
            u2 = (b2 - L[2, 0] * u0 - L[2, 1] * u1) / L[2, 2]

            # back: L^T * x = u
            x2 = u2 / L[2, 2]
            x1 = (u1 - L[2, 1] * x2) / L[1, 1]
            x0 = (u0 - L[1, 0] * x1 - L[2, 0] * x2) / L[0, 0]

            K[r, 0] = x0
            K[r, 1] = x1
            K[r, 2] = x2
        return K

    # Factored variants (accept a precomputed L) — keep NumPy impls for simplicity;
    # passing L into Numba-compiled functions often yields similar speed to NumPy BLAS here.
    def chi2_batch_factored(diff: np.ndarray, L: np.ndarray) -> np.ndarray:
        r"""
        Batched Mahalanobis :math:`\chi^2` using a precomputed Cholesky factor.

        See :func:`_chi2_batch_factored_numpy` for details.
        """
        return _chi2_batch_factored_numpy(diff, L)

    def kalman_gain_factored(P_pred: np.ndarray, H: np.ndarray, L: np.ndarray) -> np.ndarray:
        r"""
        EKF gain using a precomputed Cholesky factor :math:`L` of :math:`S`.

        See :func:`_kalman_gain_factored_numpy` for details.
        """
        return _kalman_gain_factored_numpy(P_pred, H, L)

else:
    # Pure NumPy fallbacks
    def chi2_batch(diff: np.ndarray, S: np.ndarray) -> np.ndarray:
        r"""
        Batched Mahalanobis :math:`\chi^2` via robust Cholesky (NumPy fallback).

        See Also
        --------
        _chi2_batch_numpy : Implementation details.
        """
        return _chi2_batch_numpy(diff, S)

    def kalman_gain(P_pred: np.ndarray, H: np.ndarray, S: np.ndarray) -> np.ndarray:
        r"""
        EKF gain :math:`K = P^- H^\top S^{-1}` via Cholesky solves (NumPy fallback).

        See Also
        --------
        _kalman_gain_numpy : Implementation details.
        """
        return _kalman_gain_numpy(P_pred, H, S)

    def chi2_batch_factored(diff: np.ndarray, L: np.ndarray) -> np.ndarray:
        r"""
        Batched Mahalanobis :math:`\chi^2` with precomputed :math:`L` (NumPy fallback).

        See Also
        --------
        _chi2_batch_factored_numpy : Implementation details.
        """
        return _chi2_batch_factored_numpy(diff, L)

    def kalman_gain_factored(P_pred: np.ndarray, H: np.ndarray, L: np.ndarray) -> np.ndarray:
        r"""
        EKF gain using precomputed :math:`L` (NumPy fallback).

        See Also
        --------
        _kalman_gain_factored_numpy : Implementation details.
        """
        return _kalman_gain_factored_numpy(P_pred, H, L)
