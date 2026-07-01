from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance
from vitrum.scattering import scattering


def phonon_error(dft_output: dict, mlip_output: Any) -> dict:
    """Compare phonon DOS between DFT and MLIP.

    dft_output: dict with key ``phonon_dos`` containing ``{"freq": {...}, "dos": {...}}``.
    mlip_output: atomate2 PhononBSDOSDoc with ``.phonon_dos.frequencies`` and ``.phonon_dos.densities``.
    Returns Wasserstein distance, Hellinger distance, Bhattacharyya coefficient, mean-frequency
    shift, and imaginary-mode weight for both distributions.
    """
    dft_dos = dft_output["phonon_dos"]
    mlip_dos = mlip_output.phonon_dos

    def _normalize(grid, g):
        area = np.trapezoid(g, grid)
        return g / area if area > 0 else g

    def vdos_error(f_dft, g_dft, f_mlip, g_mlip,):
        f_dft, g_dft = np.asarray(f_dft, float), np.asarray(g_dft, float)
        f_mlip, g_mlip = np.asarray(f_mlip, float), np.asarray(g_mlip, float)

        lo = min(f_dft.min(), f_mlip.min())
        hi = max(f_dft.max(), f_mlip.max())
        grid = np.linspace(lo, hi, 4000)
        gd = np.clip(np.interp(grid, f_dft, g_dft, left=0.0, right=0.0), 0, None)
        gm = np.clip(np.interp(grid, f_mlip, g_mlip, left=0.0, right=0.0), 0, None)
    
        w1 = wasserstein_distance(grid, grid, gd, gm)
    
        # --- overlap on unit-normalized densities (better-behaved than cosine) ---
        pd, pm = _normalize(grid, gd), _normalize(grid, gm)
        bc = float(np.clip(np.trapezoid(np.sqrt(pd * pm), grid), 0.0, 1.0))
        hellinger = np.sqrt(max(0.0, 1.0 - bc))
        bhattacharyya = -np.log(bc) if bc > 0 else np.inf
    
        # --- imaginary / soft-mode content: fraction of DOS weight below 0 ---
        def neg_frac(g):
            tot = np.trapezoid(g, grid)
            neg = np.trapezoid(np.where(grid < 0, g, 0.0), grid)
            return float(neg / tot) if tot > 0 else 0.0
    
        # --- first moment over *real* modes only (your shift, de-biased) ---
        def mean_pos(g):
            m = grid > 0
            return float(np.trapezoid(grid[m] * g[m], grid[m]) /
                        np.trapezoid(g[m], grid[m]))
        md, mm = mean_pos(gd), mean_pos(gm)


        return {
                "wasserstein": w1,                       # freq_unit, lower=better  (main metric)
                "hellinger": hellinger,                  # [0,1],     lower=better
                "bhattacharyya": bhattacharyya,          # >=0,       lower=better
                "bhattacharyya_coeff": bc,               # [0,1],     higher=better
                "mean_freq_dft": md,
                "mean_freq_mlip": mm,
                "mean_freq_shift": (mm - md) / md,
                "imag_weight_dft": neg_frac(gd),         # spurious-instability flag
                "imag_weight_mlip": neg_frac(gm),
                "mlip_dos": {"freq": f_mlip.tolist(), "dos": g_mlip.tolist()},
                "dft_dos": {"freq": f_dft.tolist(), "dos": g_dft.tolist()},
            }

    return vdos_error(
        np.array(list(dft_dos["freq"].values())),
        np.array(list(dft_dos["dos"].values())),
        np.array(mlip_dos.frequencies),
        np.array(mlip_dos.densities),
    )

def elastic_error(dft_output: Any, mlip_output: Any) -> dict:
    """Compare elastic moduli between DFT and MLIP.

    Both arguments are ``derived_properties`` from an atomate2 ``ElasticDoc``,
    with ``.k_vrh``, ``.g_vrh``, and ``.y_mod`` attributes.
    Returns MARE and per-modulus relative errors.
    """
    keys = ["k_vrh", "g_vrh", "y_mod"]

    output = {}

    errors = {}
    for k in keys:
        v_dft = getattr(dft_output, k)
        v_mlip = getattr(mlip_output, k)
        if v_dft is None or v_mlip is None:
            raise ValueError(f"None value encountered for key: {k}")
        errors[k] = (v_mlip - v_dft) / v_dft
        output[k] = {"mlip": v_mlip, "DFT": v_dft}

    mare = np.mean([abs(errors[k]) for k in keys])
    return {"mare": mare, "relative_errors": errors, "outputs": output}

def md_error(dft_doc: dict, mlip_output: Any) -> dict:
    """Compare partial pair distribution functions between DFT-MD and MLIP-MD.

    dft_doc: dict with keys ``r``, ``partial_pdfs`` (pair → g(r) array), and ``composition``.
    mlip_output: atomate2 MD output with ionic steps accessible via ``.output.ionic_steps``.
    Returns mean and per-pair Wasserstein distances and the g(r) arrays for both.
    """

    def _compute_rdf(trajectory):
        """Partial PDFs for a trajectory, built the same way as the DFT reference."""
        scatter = scattering(trajectory)
        r = np.array([float(x) for x in scatter.xval])
        pdfs = {
            "-".join(str(p) for p in pair): np.array([float(v) for v in pdf])
            for pair, pdf in zip(scatter.pairs, scatter.partial_pdfs)
        }
        return r, pdfs

    def _match_pair(pdfs, pair):
        """Look up a pair, tolerating reversed ordering (Si-O vs O-Si)."""
        if pair in pdfs:
            return pdfs[pair]
        return pdfs.get("-".join(reversed(pair.split("-"))))

    structures = [getattr(s, "structure", s) for s in getattr(getattr(mlip_output, "output", mlip_output), "ionic_steps", None)]

    r_mlip, pdf_mlip = _compute_rdf(structures)
    r_dft = np.asarray(dft_doc["r"])

    per_pair = {}
    mlip_pdfs = {}
    dft_pdfs = {}
    for pair, g_dft in dft_doc["partial_pdfs"].items():
        g_mlip = _match_pair(pdf_mlip, pair)
        if g_mlip is None:
            continue
        per_pair[pair] = float(
            wasserstein_distance(
                r_dft, r_mlip,
                u_weights=np.clip(np.asarray(g_dft), 0, None),
                v_weights=np.clip(g_mlip, 0, None),
            )
        )
        dft_pdfs[pair] = g_dft
        mlip_pdfs[pair] = g_mlip.tolist()

    mean = float(np.mean(list(per_pair.values()))) if per_pair else float("nan")
    return {"mean": mean,
            "per_pair": per_pair,
            "mlip_rdf": {"r": r_mlip.tolist(), "partial_pdfs": mlip_pdfs},
            "dft_rdf": {"r": r_dft.tolist(), "partial_pdfs": dft_pdfs},}