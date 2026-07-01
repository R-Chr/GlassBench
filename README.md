# GlassBench

Benchmark suite for evaluating machine learning interatomic potentials (MLIPs) against DFT reference data on glass structure properties. Covers three properties: phonon density of states, elastic moduli, and radial distribution functions from MD.

## Installation

```bash
git clone https://github.com/R-Chr/GlassBench.git
cd GlassBench
pip install -e .
```

Install with support for specific MLIP backends:

```bash
pip install -e ".[mace]"       # MACE models
pip install -e ".[grace-fs]"   # GRACE-FS models
pip install -e ".[grace]"      # GRACE models
pip install -e ".[all]"        # All backends
```

## Environment setup

There are two ways to provide a potential to GlassBench:

**Option 1 — pass an ASE calculator directly (no environment setup required):**

```python
from mace.calculators import MACECalculator
from glassbench import run_tests

calc = MACECalculator("path/to/my_model.model", device="cpu")
results = run_tests(calculator=calc, potential_name="MyMACE")
```

`potential_name` is optional here and is only used to name the output directory. If omitted the directory will be called `custom`.

Individual runners also accept a calculator directly:

```python
from glassbench import run_phonons, run_elastic, run_md

results = run_phonons(my_structures, calculator=calc)
```

**Option 2 — name-based lookup via `potentials.yaml`:**

GlassBench reads potentials from a directory you control, pointed to by the `MLIP_PARAMS_DIR` environment variable.

```bash
export MLIP_PARAMS_DIR=/path/to/your/mlip-params
```

Add this to your shell profile (`~/.bashrc`, `~/.zshrc`) or your cluster job script to make it permanent.

### potentials.yaml

`MLIP_PARAMS_DIR` must contain a `potentials.yaml` file that lists available potentials. Each entry maps a name to metadata and paths to model files (relative to `MLIP_PARAMS_DIR`, or absolute). A commented template is provided at [`mlip-params/potentials_template.yaml`](mlip-params/potentials_template.yaml).

```yaml
MACE_GlassDB:
  model_type: mace        # mace | grace | grace-fs
  functional: r2scan
  size: medium
  cpu: MACE_GlassDB.model
  gpu: MACE_GlassDB.model

GRACE_FS_2025:
  model_type: grace-fs
  functional: pbe
  size: medium
  cpu: Grace-fs-2025.yaml
  gpu: FS_PBE_model

grace_1L_r2scan:
  model_type: grace
  functional: r2scan
  size: medium
  cpu: grace_1L_r2scan
  gpu: grace_1L_r2scan
```

Supported `model_type` values and the optional dependency that must be installed for each:

| `model_type` | Backend package | Extra |
|---|---|---|
| `mace` | `mace-torch` | `.[mace]` |
| `grace-fs` | `pyace` | `.[grace-fs]` |
| `grace` | `tensorpotential` | `.[grace]` |

## Usage

### Command line

Run all three benchmarks with the default potential (`MACE_GlassDB`):

```bash
glassbench
```

Specify a different potential or output directory:

```bash
glassbench --potential GRACE_FS_2025
glassbench --potential GRACE_FS_2025 --output-dir ./my_results
```

Run a subset of benchmarks:

```bash
glassbench --potential MACE_GlassDB --no-md
glassbench --potential MACE_GlassDB --no-phonons --no-elastic
```

Full option reference:

```
glassbench --help

  --potential     Name of the potential as it appears in potentials.yaml (default: MACE_GlassDB)
  --output-dir    Directory to write results (default: ./benchmark_results/<potential>)
  --no-phonons    Skip the phonon DOS benchmark
  --no-elastic    Skip the elastic moduli benchmark
  --no-md         Skip the MD/RDF benchmark
  --fresh         Ignore existing checkpoints and start from scratch
  --verbose       Show full jobflow job-level output (useful for debugging)
```

### Python API

```python
from glassbench import run_tests

# Name-based lookup (requires MLIP_PARAMS_DIR)
results = run_tests(potential_name="MACE_GlassDB")

# Direct calculator — no environment setup needed
from mace.calculators import MACECalculator
calc = MACECalculator("my_model.model", device="cpu")
results = run_tests(calculator=calc, potential_name="MyMACE")

# Custom output location, skip MD
results = run_tests(
    potential_name="GRACE_FS_2025",
    results_dir="/scratch/benchmarks/grace",
    md=False,
)
```

The return value is a dict keyed by composition formula, each containing the error metrics for the benchmarks that were run.

## Output files

Results are written under `<output_dir>/` (one directory per potential):

```
benchmark_results/
└── MACE_GlassDB/
    ├── elastic_benchmark.json              # raw elastic workflow outputs (checkpoint)
    ├── phonon_benchmark.json               # raw phonon workflow outputs (checkpoint)
    ├── md_benchmark.json                   # raw MD workflow outputs (checkpoint)
    └── benchmark_results_MACE_GlassDB.json # aggregated error metrics
```

The three raw files double as checkpoints — each structure's result is written immediately after it completes. If the run is interrupted, restarting with the same `--output-dir` resumes from where it left off. Use `--fresh` to discard checkpoints and start from scratch.

The aggregated file has the structure:

```json
{
  "SiO2": {
    "elastic_error": { "mare": 0.05, "relative_errors": { ... }, "outputs": { ... } },
    "phonon_error": { "wasserstein": 0.12, "hellinger": 0.08, ... },
    "md_error":     { "mean": 0.03, "per_pair": { "Si-O": 0.02, ... } }
  }
}
```

Failed structures are recorded as `null` rather than omitted, making it easy to spot incomplete results. They are skipped when computing cross-composition means.

At the end of each run a summary is printed to the terminal:

```
=== Benchmark summary ===

Elastic  (MARE)
  CaAl2SiO6    0.0312
  SiO2         0.0587
  Mean         0.0450

Phonon   (Wasserstein / THz)
  CaAl2SiO6   0.2841
  SiO2         0.1923
  Mean         0.2382

MD / RDF (Wasserstein)
  CaAl2SiO6   0.0218
  SiO2         0.0341
  Mean         0.0280
```

## Benchmarks

All three workflows are built on [atomate2](https://github.com/materialsproject/atomate2). The key workflow choices are described below.

### Phonon DOS

Uses atomate2's `PhononMaker` (finite-displacement method). Compares the vibrational density of states against DFT using Wasserstein distance, Hellinger distance, and Bhattacharyya coefficient. Lower values indicate better agreement.

Key choices:
- **`sym_reduce=False`** — glass structures have no meaningful symmetry; all displacements are generated explicitly.
- **`displacement=0.01` Å** — standard finite-displacement amplitude.
- **`relax_cell=False`** — the supercell volume is fixed to the DFT reference; only atomic positions are relaxed before displacement.
- **`born_maker=None`** — non-analytical (LO-TO) correction is omitted; relevant glass compositions are predominantly non-polar.
- **`phonon_dos_sigma=0.3` THz** — Gaussian smearing applied to the DOS before comparison.
- **`min_length=1`** — no supercell expansion; the DFT reference structures are already large glass cells.

### Elastic moduli

Uses atomate2's `ElasticMaker` (strain-stress method). Compares bulk modulus ($K_\text{VRH}$), shear modulus ($G_\text{VRH}$), and Young's modulus against DFT. Reported as mean absolute relative error (MARE).

Key choices:
- **`order=2`** — second-order (linear) elastic constants.
- **`sym_reduce=False`** — no symmetry reduction of strain perturbations; appropriate for amorphous structures.
- **`fitting_method="independent"`** — each elastic constant is fitted independently rather than via a full tensor fit.
- **`relax_cell=False`** — volume fixed to the DFT reference before straining.

### MD / RDF

Uses atomate2's `ForceFieldMDMaker`. Runs a high-temperature NVT trajectory and compares partial pair distribution functions $g(r)$ to DFT-MD references via Wasserstein distance.

Key choices:
- **`ensemble="nvt"`, `dynamics="nose-hoover"`** — Nosé-Hoover thermostat for stable temperature control.
- **`temperature=3300` K** — above the liquidus for the target compositions, matching the DFT reference MD conditions.
- **`time_step=2` fs**.

## Adding a new potential

1. Place the model file(s) in `MLIP_PARAMS_DIR`.
2. Add an entry to `potentials.yaml` with `model_type`, `functional`, `size`, and the path under `cpu` (and optionally `gpu`).
3. If the `model_type` is not `mace`, `grace`, or `grace-fs`, register a builder:

```python
from glassbench import register_calculator

@register_calculator("my-mlip")
def _build_my_mlip(model_path, **kwargs):
    from my_mlip.ase import MyCalculator
    return MyCalculator(model_path, **kwargs)
```

4. Run the benchmark:

```bash
glassbench --potential my_potential_name
```
