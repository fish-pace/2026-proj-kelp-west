"""QWIP (quality water index polynomial) helpers and plots.

This module computes QWIP metrics from water-leaving reflectance spectra and
provides convenience plotting. The workflow is:

1. Subset to visible wavelengths (e.g., 400–700 nm).
2. Compute AVW (apparent visible wavelength) and integrated brightness.
3. Compute NDI(490,665) and a polynomial prediction vs AVW.
4. Classify spectra into 400A/500A/600A types and flag QWIP outliers.

Public API
----------
- :func:`calc_avw`
- :func:`calc_brightness`
- :func:`process_qwip_data`
- :func:`plot_qwip`
- :func:`plot_normalized_spectra`
- :func:`run_qwip_analysis`

"""

from __future__ import annotations

from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.integrate import trapezoid

__all__ = [
    "QwipResults",
    "calc_avw",
    "calc_brightness",
    "process_qwip_data",
    "plot_qwip",
    "plot_normalized_spectra",
    "run_qwip_analysis",
]


class QwipResults(TypedDict):
    """Container for QWIP outputs from :func:`process_qwip_data`."""

    avw: np.ndarray
    NDI: np.ndarray
    QWIP_Score: np.ndarray
    ind_400A: np.ndarray  # bool mask
    ind_500A: np.ndarray  # bool mask
    ind_600A: np.ndarray  # bool mask
    poly_coeffs: np.ndarray
    failed_indices: np.ndarray  # int idx array
    passed_indices: np.ndarray  # int idx array


def calc_avw(wavelengths: np.ndarray, rrs: np.ndarray) -> np.ndarray:
    """Calculate Apparent Visible Wavelength (AVW).

    AVW is defined here as::

        AVW = sum(Rrs) / sum(Rrs / lambda)

    Parameters
    ----------
    wavelengths
        1-D array of wavelengths (nm) of shape ``(n_wl,)``.
    rrs
        Water-leaving reflectance array of shape ``(n_samples, n_wl)`` or
        ``(n_wl,)``. If 1-D, it is treated as one sample.

    Returns
    -------
    numpy.ndarray
        AVW per sample, shape ``(n_samples,)``.

    Raises
    ------
    ValueError
        If the wavelengths length does not match the reflectance spectral axis.

    """
    wl = np.asarray(wavelengths, dtype=float).reshape(-1)
    X = np.atleast_2d(np.asarray(rrs, dtype=float))
    if X.shape[1] != wl.shape[0]:
        raise ValueError("wavelengths length must match rrs second dimension")

    num = np.sum(X, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.sum(X / wl, axis=1)
        avw = num / den
    return avw


def calc_brightness(wavelengths: np.ndarray, rrs: np.ndarray) -> np.ndarray:
    """Calculate integrated “brightness” using the trapezoidal rule.

    Parameters
    ----------
    wavelengths
        1-D array of wavelengths (nm) of shape ``(n_wl,)``.
    rrs
        Reflectance array of shape ``(n_samples, n_wl)`` or ``(n_wl,)``.

    Returns
    -------
    numpy.ndarray
        Integral per sample, shape ``(n_samples,)``.

    Raises
    ------
    ValueError
        If the wavelength length does not match the reflectance spectral axis.

    """
    wl = np.asarray(wavelengths, dtype=float).reshape(-1)
    X = np.atleast_2d(np.asarray(rrs, dtype=float))
    if X.shape[1] != wl.shape[0]:
        raise ValueError("wavelengths length must match rrs second dimension")
    return trapezoid(y=X, x=wl, axis=-1)


def process_qwip_data(
    wavelengths: np.ndarray,
    rrs: np.ndarray,
    avw: np.ndarray,
) -> QwipResults:
    """Compute QWIP masks and scores for spectra.

    Parameters
    ----------
    wavelengths
        1-D array of wavelengths (nm), shape ``(n_wl,)``.
    rrs
        Reflectance array of shape ``(n_samples, n_wl)``.
    avw
        Apparent Visible Wavelength per sample, shape ``(n_samples,)``.

    Returns
    -------
    QwipResults
        Typed mapping with AVW, NDI, QWIP_Score, type masks, and pass/fail indices.

    Notes
    -----
    - NDI is computed as ``(Rrs665 - Rrs490) / (Rrs665 + Rrs490)``.
    - The polynomial coefficients are fixed (Balasubramanian 2020 fit).
    - Fail criteria: type-range violations OR ``|QWIP_Score| > 0.2``.

    """
    wl = np.asarray(wavelengths, dtype=float).reshape(-1)
    X = np.asarray(rrs, dtype=float)
    if X.ndim != 2:
        raise ValueError("rrs must be a 2-D array (n_samples, n_wl)")
    if X.shape[1] != wl.shape[0]:
        raise ValueError("wavelengths length must match rrs second dimension")
    avw = np.asarray(avw, dtype=float).reshape(-1)
    if avw.shape[0] != X.shape[0]:
        raise ValueError("avw length must match number of spectra in rrs")

    # Find nearest indices to target wavelengths
    idx490 = int(np.argmin(np.abs(wl - 490.0)))
    idx560 = int(np.argmin(np.abs(wl - 560.0)))
    idx665 = int(np.argmin(np.abs(wl - 665.0)))

    # Normalized Difference Index (NDI)
    num = X[:, idx665] - X[:, idx490]
    den = X[:, idx665] + X[:, idx490]
    with np.errstate(invalid="ignore", divide="ignore"):
        ndi = num / den

    # Polynomial coefficients for NDI vs AVW
    p = np.array(
        [-8.399885e-09, 1.715532e-05, -1.301670e-02, 4.357838, -5.449532e02],
        dtype=float,
    )

    # Type masks (Balasubramanian 2020)
    step1 = X[:, idx665] > X[:, idx560]
    step2 = X[:, idx665] > 0.025
    step3 = X[:, idx560] < X[:, idx490]

    ind_600A = step1 | step2
    ind_500A = (~step1 & ~step2) & ~step3
    ind_400A = (~step1 & ~step2) & step3

    # QWIP score (difference from polynomial prediction)
    ndi_pred = np.polyval(p, avw)
    qwip_score = ndi - ndi_pred

    # --- Classification Logic ---

    # Fail flags
    failed_400 = ind_400A & ((avw < 410.0) | (avw > 520.0))
    failed_500 = ind_500A & ((avw < 490.0) | (avw > 590.0))
    failed_600 = ind_600A & ((avw < 550.0) | (avw > 600.0))
    type_failure = failed_400 | failed_500 | failed_600
    qwip_failure = (qwip_score < -0.2) | (qwip_score > 0.2)

    is_failed = type_failure | qwip_failure
    failed_indices = np.where(is_failed)[0]
    passed_indices = np.where(~is_failed)[0]

    return QwipResults(
        avw=avw,
        NDI=ndi,
        QWIP_Score=qwip_score,
        ind_400A=ind_400A,
        ind_500A=ind_500A,
        ind_600A=ind_600A,
        poly_coeffs=p,
        failed_indices=failed_indices,
        passed_indices=passed_indices,
    )


def plot_qwip(
    results: QwipResults,
    title: str = "QWIP Analysis",
    *,
    show: bool = True,
):
    """Generate a QWIP scatter with polynomial bands.

    Parameters
    ----------
    results
        Output from :func:`process_qwip_data`.
    title
        Plot title.
    show
        If ``True`` (default), call :func:`matplotlib.pyplot.show`. If ``False``,
        the caller is responsible for drawing/closing the figure.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axis.

    """
    avw = results["avw"]
    ndi = results["NDI"]
    ind_400A = results["ind_400A"]
    ind_500A = results["ind_500A"]
    ind_600A = results["ind_600A"]
    p = results["poly_coeffs"]

    avw_poly = np.arange(400.0, 631.0)
    fit1 = np.polyval(p, avw_poly)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title, fontsize=18, fontweight="bold")

    # Classified points
    ax.plot(
        avw[ind_400A],
        ndi[ind_400A],
        "ok",
        markersize=5,
        markerfacecolor="b",
        label="Type I Blue-Green",
    )
    ax.plot(
        avw[ind_500A],
        ndi[ind_500A],
        "ok",
        markersize=5,
        markerfacecolor="g",
        label="Type II Green",
    )
    ax.plot(
        avw[ind_600A],
        ndi[ind_600A],
        "ok",
        markersize=5,
        markerfacecolor="r",
        label="Type III Brown",
    )

    # Polynomial & thresholds
    ax.plot(avw_poly, fit1, "-k", linewidth=2)
    ax.plot(avw_poly, fit1 + 0.1, "--g", linewidth=2)
    ax.plot(avw_poly, fit1 - 0.1, "--g", linewidth=2)
    ax.plot(avw_poly, fit1 + 0.2, "--", linewidth=2, color=(0.9290, 0.6940, 0.1250))
    ax.plot(avw_poly, fit1 - 0.2, "--", linewidth=2, color=(0.9290, 0.6940, 0.1250))
    ax.plot(avw_poly, fit1 + 0.3, "--", linewidth=2, color=(0.8500, 0.3250, 0.0980))
    ax.plot(avw_poly, fit1 - 0.3, "--", linewidth=2, color=(0.8500, 0.3250, 0.0980))
    ax.plot(avw_poly, fit1 + 0.4, "-r", linewidth=2)
    ax.plot(avw_poly, fit1 - 0.4, "-r", linewidth=2)

    # Labels and legends
    ax.set_xlabel("AVW (nm)", fontsize=16)
    ax.set_ylabel("NDI (490, 665)", fontsize=16)
    ax.set_ylim(-2.5, 2.0)
    ax.set_xlim(440, 600)
    ax.grid(True, linestyle=":", alpha=0.6)

    legend1 = ax.legend(loc="lower right", title="Data Types", fontsize=12)
    ax.add_artist(legend1)

    legend_lines = [
        Line2D([0], [0], color="g", lw=2, linestyle="--", label="QWIP ± 0.1"),
        Line2D(
            [0],
            [0],
            color=(0.9290, 0.6940, 0.1250),
            lw=2,
            linestyle="--",
            label="QWIP ± 0.2",
        ),
        Line2D(
            [0],
            [0],
            color=(0.8500, 0.3250, 0.0980),
            lw=2,
            linestyle="--",
            label="QWIP ± 0.3",
        ),
        Line2D([0], [0], color="r", lw=2, linestyle="-", label="QWIP ± 0.4"),
    ]
    ax.legend(handles=legend_lines, loc="upper left", title="Thresholds", fontsize=12)

    if show:
        plt.show()
    return fig, ax


def plot_normalized_spectra(
    results: QwipResults,
    wavelengths: np.ndarray,
    rrs: np.ndarray,
    brightness: np.ndarray,
    *,
    title_suffix: str = "",
    show: bool = True,
):
    """Plot brightness-normalized spectra for pass/fail sets, colored by AVW.

    Parameters
    ----------
    results
        Output from :func:`process_qwip_data`.
    wavelengths
        1-D wavelengths (nm), shape ``(n_wl,)``.
    rrs
        Reflectance array, shape ``(n_samples, n_wl)``.
    brightness
        Integrated brightness per sample, from :func:`calc_brightness`.
    title_suffix
        Appended to the figure title.
    show
        If ``True`` (default) call ``plt.show()``; otherwise just return fig/axes.

    Returns
    -------
    (fig, (ax1, ax2))
        Matplotlib figure and axes for passed/failed spectra.

    """
    wl = np.asarray(wavelengths, dtype=float).reshape(-1)
    X = np.asarray(rrs, dtype=float)
    if X.ndim != 2 or X.shape[1] != wl.shape[0]:
        raise ValueError("wavelengths length must match rrs second dimension")
    brightness_arr = np.asarray(brightness, dtype=float).reshape(-1)
    if brightness_arr.shape[0] != X.shape[0]:
        raise ValueError("brightness length must match number of spectra in rrs")

    with np.errstate(divide="ignore", invalid="ignore"):
        norm_rrs = X / brightness_arr[:, None]
    norm_rrs[~np.isfinite(norm_rrs)] = np.nan
    passed = results["passed_indices"]
    failed = results["failed_indices"]
    avw = results["avw"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle(f"Normalized Spectra - {title_suffix}", fontsize=18, fontweight="bold")

    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(vmin=440, vmax=600)

    # Passed
    ax1.set_title(f"Passed Spectra ({len(passed)})", fontsize=14)
    for i in passed:
        ax1.plot(wl, norm_rrs[i, :], color=cmap(norm(avw[i])), alpha=0.6)
    ax1.set_xlabel("Wavelength (nm)", fontsize=12)
    ax1.set_ylabel("Normalized Rrs (Rrs / Brightness)", fontsize=12)