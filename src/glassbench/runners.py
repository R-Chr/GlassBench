from __future__ import annotations

import importlib.resources
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jobflow import run_locally
from monty.serialization import MontyDecoder, MontyEncoder

from glassbench.jobs import make_elastic_mlip, make_MD_mlip, make_phonon_mlip
from glassbench.metrics import elastic_error, md_error, phonon_error

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


def _failed(result) -> bool:
    """Failed runs are stored as the error message (None in older checkpoints)."""
    return result is None or isinstance(result, str)


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f, cls=MontyDecoder)


def _save_checkpoint(path: Path, data: dict) -> None:
    with path.open("w") as f:
        json.dump(data, f, cls=MontyEncoder, indent=2)


def _run_workflow(
    structures: list[Structure],
    make_fn,
    potential_kwargs: dict | None,
    checkpoint_file: Path | None = None,
    verbose: bool = False,
    calculator: Any = None,
    **kwargs,
) -> dict:
    results_dict = _load_checkpoint(checkpoint_file) if checkpoint_file is not None else {}

    # retry failures; only successful runs count as done
    pending = [s for s in structures if _failed(results_dict.get(s.composition.reduced_formula))]
    n_total = len(structures)
    n_done = n_total - len(pending)

    if n_done:
        logger.info("Resuming: %d/%d already complete, skipping", n_done, n_total)

    for i, structure in enumerate(pending):
        formula = structure.composition.reduced_formula
        logger.info("[%d/%d] Starting %s", n_done + i + 1, n_total, formula)
        try:
            workflow = make_fn(structure, potential_kwargs, calculator=calculator, **kwargs)
            response = run_locally(workflow, log=verbose, create_folders=False, ensure_success=True, raise_immediately=True)
            if make_fn in (make_phonon_mlip, make_elastic_mlip):
                results_dict[formula] = response[workflow.jobs[-1].uuid][1].output
            elif make_fn is make_MD_mlip:
                results_dict[formula] = response[workflow.uuid][1].output
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logger.warning("FAILED %s: %s", formula, msg, exc_info=verbose)
            results_dict[formula] = msg  # failure sentinel, see _failed()

        if checkpoint_file is not None:
            _save_checkpoint(checkpoint_file, results_dict)

    return results_dict


def run_phonons(structures: list[Structure], potential_kwargs: dict | None = None,
                checkpoint_file: Path | None = None, verbose: bool = False,
                calculator: Any = None) -> dict:
    return _run_workflow(structures, make_phonon_mlip, potential_kwargs,
                         checkpoint_file=checkpoint_file, verbose=verbose,
                         calculator=calculator, force_tol=0.02)

def run_elastic(structures: list[Structure], potential_kwargs: dict | None = None,
                checkpoint_file: Path | None = None, verbose: bool = False,
                calculator: Any = None) -> dict:
    return _run_workflow(structures, make_elastic_mlip, potential_kwargs,
                         checkpoint_file=checkpoint_file, verbose=verbose,
                         calculator=calculator, force_tol=0.02)

def run_md(structures: list[Structure], potential_kwargs: dict | None = None,
           TEBEG: int = 3300, n_steps: int = 1000,
           checkpoint_file: Path | None = None, verbose: bool = False,
           calculator: Any = None) -> dict:
    return _run_workflow(structures, make_MD_mlip, potential_kwargs,
                         checkpoint_file=checkpoint_file, verbose=verbose,
                         calculator=calculator, TEBEG=TEBEG, n_steps=n_steps)


def _data_file(name):
    return importlib.resources.files("glassbench.data") / name


def run_tests(
    potential_name: str | None = None,
    results_dir: str | Path | None = None,
    phonons: bool = True,
    elastic: bool = True,
    md: bool = True,
    fresh: bool = False,
    verbose: bool = False,
    calculator: Any = None,
    architecture: str = "cpu",
    device: str | None = None,
):
    if calculator is None and potential_name is None:
        raise ValueError("Provide either potential_name or calculator.")

    potential_kwargs = None
    if calculator is None:
        potential_kwargs = {"potential_name": potential_name, "architecture": architecture}
        if device:
            potential_kwargs["builder_kwargs"] = {"device": device}
    run_label = potential_name or "custom"
    benchmark_results = defaultdict(dict)

    out = Path(results_dir) if results_dir is not None else Path.cwd() / "benchmark_results" / run_label
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Potential : %s", run_label)
    logger.info("Output    : %s", out)

    if fresh:
        for name in ("elastic_benchmark.json", "phonon_benchmark.json", "md_benchmark.json"):
            (out / name).unlink(missing_ok=True)
        logger.info("Checkpoints cleared (--fresh)")

    if elastic:
        logger.info("--- Elastic benchmark ---")
        with _data_file("elastic_benchmark_dft.json").open("r") as f:
            elastic_docs = MontyDecoder().process_decoded(json.load(f))

        elastic_structures = [elastic_docs[doc].structure for doc in elastic_docs]
        elastic_results = run_elastic(elastic_structures, potential_kwargs,
                                      checkpoint_file=out / "elastic_benchmark.json",
                                      verbose=verbose, calculator=calculator)

        n_ok = sum(not _failed(v) for v in elastic_results.values())
        logger.info("Elastic done: %d/%d succeeded", n_ok, len(elastic_results))
        for formula, mlip_result in elastic_results.items():
            if _failed(mlip_result):
                benchmark_results[formula]["elastic_error"] = mlip_result
                continue
            dft_result = elastic_docs[f"Bench-elastic-glass {formula}"]
            benchmark_results[formula]["elastic_error"] = elastic_error(
                dft_result.derived_properties, mlip_result.derived_properties
            )

    if phonons:
        logger.info("--- Phonon benchmark ---")
        with _data_file("phonon_benchmark_dft.json").open("r") as f:
            phonon_docs = json.load(f)

        phonon_structures = [MontyDecoder().process_decoded(phonon_docs[doc]["structure"]) for doc in phonon_docs]
        phonon_results = run_phonons(phonon_structures, potential_kwargs,
                                     checkpoint_file=out / "phonon_benchmark.json",
                                     verbose=verbose, calculator=calculator)

        n_ok = sum(not _failed(v) for v in phonon_results.values())
        logger.info("Phonon done: %d/%d succeeded", n_ok, len(phonon_results))
        for formula, mlip_result in phonon_results.items():
            if _failed(mlip_result):
                benchmark_results[formula]["phonon_error"] = mlip_result
                continue
            benchmark_results[formula]["phonon_error"] = phonon_error(phonon_docs[formula], mlip_result)

    if md:
        logger.info("--- MD/RDF benchmark ---")
        with _data_file("rdf_benchmark_dft.json").open("r") as f:
            rdf_docs = json.load(f, cls=MontyDecoder)

        dft_results = rdf_docs["results"]
        dft_by_formula = {doc["composition"]: doc for doc in dft_results.values()}
        start_structures = [doc["start_structure"] for doc in dft_results.values()]
        md_results = run_md(start_structures, potential_kwargs,
                            checkpoint_file=out / "md_benchmark.json",
                            verbose=verbose, calculator=calculator)

        n_ok = sum(not _failed(v) for v in md_results.values())
        logger.info("MD done: %d/%d succeeded", n_ok, len(md_results))
        for formula, mlip_result in md_results.items():
            if _failed(mlip_result):
                benchmark_results[formula]["md_error"] = mlip_result
                continue
            dft_doc = dft_by_formula.get(formula)
            if dft_doc is None:
                logger.warning("No DFT reference for %s, skipping", formula)
                continue
            benchmark_results[formula]["md_error"] = md_error(dft_doc, mlip_result)

    summary_path = out / f"benchmark_results_{run_label}.json"
    with summary_path.open("w") as f:
        json.dump(benchmark_results, f, indent=4)

    _log_summary(benchmark_results)
    logger.info("Results written to %s", summary_path)

    return benchmark_results


def _log_summary(benchmark_results: dict) -> None:
    formulas = list(benchmark_results)
    if not formulas:
        return

    col = max(len(f) for f in formulas)

    def _section(title: str, key: str, metric_fn, unit: str) -> None:
        rows = {f: benchmark_results[f].get(key) for f in formulas if key in benchmark_results[f]}
        if not rows:
            return
        logger.info("")
        logger.info("%s  (%s)", title, unit)
        values = []
        for formula, result in rows.items():
            if _failed(result):
                reason = (result or "unknown error").splitlines()[0][:100]
                logger.info("  %-*s  FAILED  %s", col, formula, reason)
            else:
                v = metric_fn(result)
                logger.info("  %-*s  %.4f", col, formula, v)
                values.append(v)
        if values:
            import numpy as np
            logger.info("  %-*s  %.4f", col, "Mean", float(np.mean(values)))

    logger.info("")
    logger.info("=== Benchmark summary ===")
    _section("Elastic  ", "elastic_error", lambda r: r["mare"],         "MARE")
    _section("Phonon   ", "phonon_error",  lambda r: r["wasserstein"],  "Wasserstein / THz")
    _section("MD / RDF ", "md_error",      lambda r: r["mean"],         "Wasserstein")


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run GlassBench MLIP benchmarks")
    parser.add_argument("--potential", default="MACE_GlassDB", help="Potential name from potentials.yaml")
    parser.add_argument("--output-dir", default=None, help="Directory for results (default: ./benchmark_results/<potential>)")
    parser.add_argument("--no-phonons", action="store_true", help="Skip phonon benchmark")
    parser.add_argument("--no-elastic", action="store_true", help="Skip elastic benchmark")
    parser.add_argument("--no-md", action="store_true", help="Skip MD/RDF benchmark")
    parser.add_argument("--architecture", default="cpu", help="Which model file to load from potentials.yaml (cpu/gpu)")
    parser.add_argument("--device", default=None, help="Torch device passed to the calculator, e.g. cuda")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints and start from scratch")
    parser.add_argument("--verbose", action="store_true", help="Show full jobflow job-level output")
    args = parser.parse_args()

    # Glassbench gets its own clean handler and does NOT propagate to the root
    # logger, so its messages are never duplicated by any handler a library
    # may have added to the root logger at import time.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    gb = logging.getLogger("glassbench")
    gb.setLevel(logging.INFO)
    gb.addHandler(handler)
    gb.propagate = False

    run_tests(
        potential_name=args.potential,
        results_dir=args.output_dir,
        phonons=not args.no_phonons,
        elastic=not args.no_elastic,
        md=not args.no_md,
        fresh=args.fresh,
        verbose=args.verbose,
        architecture=args.architecture,
        device=args.device,
    )


if __name__ == "__main__":
    main()
