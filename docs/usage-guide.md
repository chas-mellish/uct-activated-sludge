# UCT Activated Sludge Model — Usage Guide

This guide covers installation, CLI usage, configuration file format, Python API, and end-to-end workflow examples for the UCT Activated Sludge Model (`uct-asp`).

## Table of Contents

- [Installation](#installation)
- [CLI Command Reference](#cli-command-reference)
  - [steady-state](#steady-state)
  - [diurnal](#diurnal)
  - [params](#params)
- [Configuration File Format (TOML)](#configuration-file-format-toml)
  - [kinetics](#kinetics)
  - [stoichiometry](#stoichiometry)
  - [wastewater](#wastewater)
  - [plant](#plant)
  - [integration](#integration)
- [Diurnal Input Data Format](#diurnal-input-data-format)
  - [CSV Format](#csv-format)
  - [TOML Format](#toml-format)
- [Python API](#python-api)
  - [Loading Configuration](#loading-configuration)
  - [Running a Steady-State Simulation](#running-a-steady-state-simulation)
  - [Running a Diurnal Simulation](#running-a-diurnal-simulation)
  - [Inspecting Parameters](#inspecting-parameters)
- [Output Format](#output-format)
  - [Steady-State Output](#steady-state-output)
  - [Diurnal Output](#diurnal-output)
  - [ASM1 Compound Index](#asm1-compound-index)
- [Workflow Examples](#workflow-examples)
  - [1. Steady-State with Default Configuration](#1-steady-state-with-default-configuration)
  - [2. Steady-State with Custom Configuration](#2-steady-state-with-custom-configuration)
  - [3. Diurnal Simulation with Flow Pattern](#3-diurnal-simulation-with-flow-pattern)
  - [4. Parameter Inspection](#4-parameter-inspection)
  - [5. Settled Sewage Simulation](#5-settled-sewage-simulation)
- [Troubleshooting](#troubleshooting)

---

## Installation

Requires Python 3.10 or later. Install in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

This installs the package and registers the `uct-asp` CLI command. Core dependencies (NumPy, SciPy, pandas) are installed automatically. Dev dependencies include pytest, Hypothesis, and matplotlib.

Verify the installation:

```bash
uct-asp --version
```

Expected output: `uct-asp 0.1.0`

---

## CLI Command Reference

```
uct-asp [--version] {steady-state,diurnal,params} ...
```

Running `uct-asp` with no arguments prints the help message.

### steady-state

Run a steady-state BNR simulation.

```bash
uct-asp steady-state [--config PATH] [--output PATH]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | `None` | Path to a TOML configuration file. If omitted, uses built-in defaults. |
| `--output` | `str` | `None` | Path to write JSON results. If omitted, results are printed to stdout as formatted text only. |

The steady-state simulation solves for the equilibrium concentrations of all 13 ASM1 compounds across all reactors using Newton-Raphson iteration. It prints a formatted text report to stdout showing compound concentrations, oxygen uptake rates, and denitrification rates.

### diurnal

Run a diurnal (dynamic) simulation.

```bash
uct-asp diurnal [--config PATH] [--output PATH] [--diurnal-data PATH]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | `None` | Path to a TOML configuration file. If omitted, uses built-in defaults. |
| `--output` | `str` | `None` | Path to write JSON results. |
| `--diurnal-data` | `str` | `None` | Path to a diurnal input data file (CSV or TOML). If omitted, uses a flat/constant pattern matching the plant's average flow, COD, and TKN. |

The diurnal simulation first runs a steady-state solve to establish initial conditions, then cycles through 24-hour periods with 12 two-hourly influent intervals until periodic steady state is reached (Xbh and Sbp converge within 0.7% between start and end of a cycle).

### params

Display default parameter values for a given parameter group.

```bash
uct-asp params --list GROUP
```

| Flag | Type | Choices | Description |
|------|------|---------|-------------|
| `--list` | `str` | `kinetics`, `stoichiometry`, `wastewater`, `plant`, `integration` | Parameter group to display. Required. |

Prints a table showing each field name, its type, and the default value for the selected parameter group.

---

## Configuration File Format (TOML)

Plant configurations are specified in TOML files with five sections. All keys are optional — omitted keys use the built-in defaults from the Python dataclasses. See `examples/default_plant.toml` for a complete annotated example.

### kinetics

Kinetic rate constants and Arrhenius temperature-correction factors. Supports flat layout or nested sub-tables.

**Flat layout:**

```toml
[kinetics]
MuHatHetero20 = 3.2
Ks20 = 5.0
```

**Nested layout (recommended for readability):**

```toml
[kinetics.heterotrophs]
MuHatHetero20 = 3.2
Ks20          = 5.0

[kinetics.autotrophs]
MuHatAuto20 = 0.45

[kinetics.arrhenius]
ThetaMuHatH = 1.200
```

#### Heterotrophic parameters (20°C reference)

| Key | Type | Default | Unit | Description |
|-----|------|---------|------|-------------|
| `MuHatHetero20` | float | 3.2 | d⁻¹ | Maximum specific growth rate |
| `Ks20` | float | 5.0 | g COD m⁻³ | Half-saturation for Ss |
| `Bh20` | float | 0.62 | d⁻¹ | Endogenous respiration rate |
| `Koh` | float | 0.002 | g O₂ m⁻³ | Half-saturation for DO (heterotrophs) |
| `NetaGrow` | float | 0.33 | — | Anoxic growth reduction factor |
| `Kno` | float | 0.10 | g N m⁻³ | Half-saturation for NO₃ |
| `Kmp20` | float | 1.35 | d⁻¹ | Maximum rate for Xp hydrolysis |
| `Ksp20` | float | 0.027 | g COD / g COD | Half-saturation for Xp hydrolysis |
| `Kr20` | float | 0.032 | d⁻¹ | Adsorption rate constant |
| `Kna` | float | 0.01 | g N m⁻³ | Half-saturation for NH₃ |
| `Ka20` | float | 0.17 | m³ g⁻¹ COD d⁻¹ | Ammonification rate constant |

#### Autotrophic parameters (20°C reference)

| Key | Type | Default | Unit | Description |
|-----|------|---------|------|-------------|
| `MuHatAuto20` | float | 0.45 | d⁻¹ | Maximum specific growth rate |
| `Knh20` | float | 1.0 | g N m⁻³ | Half-saturation for NH₃ |
| `Koa` | float | 0.002 | g O₂ m⁻³ | Half-saturation for DO (autotrophs) |
| `Ba20` | float | 0.04 | d⁻¹ | Endogenous respiration rate |

#### Arrhenius temperature-correction factors

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ThetaMuHatH` | float | 1.200 | Theta for MuHatHetero |
| `ThetaKs` | float | 1.000 | Theta for Ks (no temperature effect) |
| `ThetaBh` | float | 1.029 | Theta for Bh |
| `ThetaKmp` | float | 1.08 | Theta for Kmp |
| `ThetaKsp` | float | 0.910 | Theta for Ksp |
| `ThetaKr` | float | 1.029 | Theta for Kr |
| `ThetaKa` | float | 1.029 | Theta for Ka |
| `ThetaMuHatA` | float | 1.123 | Theta for MuHatAuto |
| `ThetaKnh` | float | 1.123 | Theta for Knh |
| `ThetaBa` | float | 1.029 | Theta for Ba |

Temperature correction is applied as: `param(T) = param20 * theta^(T - 20)`

### stoichiometry

Yield and stoichiometric coefficients.

```toml
[stoichiometry]
Yh     = 0.666
Fe     = 0.08
Ixb    = 0.0676
Ixe    = 0.0676
Ya     = 0.15
CODVSS = 1.48
Fma    = 1.00
```

| Key | Type | Default | Unit | Description |
|-----|------|---------|------|-------------|
| `Yh` | float | 0.666 | g COD / g COD | Heterotrophic yield (0.45 × 1.48) |
| `Fe` | float | 0.08 | — | Endogenous residue fraction |
| `Ixb` | float | 0.0676 | g N / g COD | N content of biomass (0.1 / 1.48) |
| `Ixe` | float | 0.0676 | g N / g COD | N content of endogenous residue |
| `Ya` | float | 0.15 | g COD / g N | Autotrophic yield |
| `CODVSS` | float | 1.48 | g COD / g VSS | COD-to-VSS ratio |
| `Fma` | float | 1.00 | — | Maximum aerobic fraction |

### wastewater

Influent wastewater composition and COD/TKN fractionation parameters.

```toml
[wastewater]
Sti     = 500.0
Nti     = 50.0
Fbs     = 0.20
Fus     = 0.05
Fup     = 0.13
Fnaa    = 0.75
Fnox    = 0.50
Fnu     = 0.03
Fxbh    = 0.0
VSSTSS  = 0.75
Alki    = 10.0
settled = false
```

| Key | Type | Default | Unit | Description |
|-----|------|---------|------|-------------|
| `Sti` | float | 500.0 | g COD m⁻³ | Total influent COD |
| `Nti` | float | 50.0 | g N m⁻³ | Total influent TKN |
| `Fbs` | float | 0.20 | — | Readily biodegradable fraction of Sbi |
| `Fus` | float | 0.05 | — | Unbiodegradable soluble fraction of Sti |
| `Fup` | float | 0.13 | — | Unbiodegradable particulate fraction of Sti |
| `Fnaa` | float | 0.75 | — | Free-and-saline ammonia fraction of OrgN |
| `Fnox` | float | 0.50 | — | Biodegradable fraction of organic N |
| `Fnu` | float | 0.03 | — | Unbiodegradable fraction of soluble OrgN |
| `Fxbh` | float | 0.0 | — | Influent active heterotroph fraction |
| `VSSTSS` | float | 0.75 | g VSS / g TSS | VSS-to-TSS ratio |
| `Alki` | float | 10.0 | meq/L as CaCO₃ | Influent alkalinity |
| `settled` | bool | `false` | — | Whether settled-sewage fractionation is active |

For settled sewage, typical values are: `Fbs=0.25`, `Fus=0.08`, `Fup=0.04`, `Fnaa=0.83`, `Fnu=0.04`, `VSSTSS=0.83`, `settled=true`. See `examples/settled_sewage.toml`.

### plant

Reactor topology and operating parameters.

```toml
[plant]
LastReactor = 3
Temp        = 20.0
Rs          = 20.0
VolType     = "ML"
```

#### Scalar parameters

| Key | Type | Default | Unit | Description |
|-----|------|---------|------|-------------|
| `LastReactor` | int | 3 | — | Number of active reactors (1–12) |
| `Temp` | float | 20.0 | °C | Operating temperature |
| `Rs` | float | 20.0 | days | Sludge retention time (SRT) |
| `VolType` | str | `"ML"` | — | Volume unit label (must be `"ML"`) |
| `FlowFeed` | float | 25.0 | ML/d | Influent flow rate |
| `FlowRASrecycle` | float | 25.0 | ML/d | Return activated sludge recycle flow |
| `FlowArecycle` | float | 75.0 | ML/d | A-recycle flow rate |
| `FlowBrecycle` | float | 0.0 | ML/d | B-recycle flow rate |
| `FlowWaste` | float | 0.0 | ML/d | Wastage flow rate (0 = auto-calculated) |
| `ReactorAIn` | int | 2 | — | A-recycle destination reactor number |
| `ReactorAOut` | int | 3 | — | A-recycle source reactor number |
| `ReactorBIn` | int | 0 | — | B-recycle destination reactor number |
| `ReactorBOut` | int | 0 | — | B-recycle source reactor number |

#### Array parameters

All arrays are 1-indexed TOML lists with up to 13 entries (positions 1–12 correspond to reactors, position 13 is the settling tank). Unused positions should be 0 (or `false` for boolean arrays).

| Key | Type | Description |
|-----|------|-------------|
| `Vol` | float[] | Reactor volumes (ML) |
| `FracFeed` | float[] | Fractional feed distribution to each reactor (must sum to 1.0) |
| `DOConc` | float[] | Dissolved oxygen setpoint per reactor (g O₂ m⁻³) |
| `ReactorAerated` | bool[] | Whether each reactor is aerated (`true`/`false`) |
| `FlagRASIn` | int[] | RAS recycle routing flags (1 = connected to that reactor) |
| `FlagAIn` | int[] | A-recycle inflow routing flags |
| `FlagAOut` | int[] | A-recycle outflow routing flags |

Example for a 3-reactor UCT process (anaerobic → anoxic → aerobic):

```toml
Vol = [1.5, 3.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
FracFeed = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DOConc = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ReactorAerated = [false, false, true, true, true, true, true, true, true, true, true, true, true]
```

### integration

Numerical integration control parameters for the diurnal simulation.

```toml
[integration]
Accuracy     = 0.50
Theta        = 0.70
DataIntHours = 0.25
```

| Key | Type | Default | Unit | Description |
|-----|------|---------|------|-------------|
| `Accuracy` | float | 0.50 | — | Relative error tolerance (fraction) |
| `Theta` | float | 0.70 | — | Implicit/explicit weighting (0 = explicit, 1 = fully implicit) |
| `DataIntHours` | float | 0.25 | hours | Data interpolation interval (0.25 = 15 minutes) |

---

## Diurnal Input Data Format

The diurnal simulation requires a 12-record input file with 2-hourly influent data. Each record specifies the flow rate, COD, and TKN for a 2-hour interval of the day.

### CSV Format

A CSV file with a header row and 12 data rows:

```csv
Time,Flow,COD,TKN
0,12.5,400.0,40.0
2,10.0,350.0,35.0
4,12.5,380.0,38.0
6,22.5,520.0,52.0
8,37.5,650.0,65.0
10,32.5,580.0,58.0
12,27.5,500.0,50.0
14,25.0,480.0,48.0
16,27.5,510.0,51.0
18,35.0,620.0,62.0
20,30.0,550.0,55.0
22,17.5,460.0,46.0
```

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `Time` | float | hours | Time of day (0, 2, 4, ..., 22) |
| `Flow` | float | ML/d | Influent flow rate during this interval |
| `COD` | float | g COD m⁻³ | Influent total COD |
| `TKN` | float | g N m⁻³ | Influent TKN |

See `examples/diurnal_pattern.csv` for a realistic municipal wastewater pattern.

### TOML Format

A TOML file using an array-of-tables with `[[diurnal]]`:

```toml
[[diurnal]]
Time = 0.0
Flow = 12.5
COD = 400.0
TKN = 40.0

[[diurnal]]
Time = 2.0
Flow = 10.0
COD = 350.0
TKN = 35.0

# ... repeat for all 12 intervals
```

---

## Python API

The package can be used programmatically without the CLI.

### Loading Configuration

```python
from uct_activated_sludge.config import load_config, build_params_from_config

# Load from a TOML file
config = load_config("examples/default_plant.toml")
kp, sp, wp, pc, ip = build_params_from_config(config)

# Or use defaults (empty dict)
kp, sp, wp, pc, ip = build_params_from_config({})
```

`build_params_from_config()` returns a 5-tuple:
- `kp` — `KineticParams` (kinetic rate constants)
- `sp` — `StoichiometricParams` (yield coefficients)
- `wp` — `WastewaterParams` (influent fractionation)
- `pc` — `PlantConfig` (reactor topology and flows)
- `ip` — `IntegrationParams` (numerical integration control)

### Running a Steady-State Simulation

```python
from uct_activated_sludge.config import load_config, build_params_from_config
from uct_activated_sludge.steady_state import run_steady_state

config = load_config("examples/default_plant.toml")
kp, sp, wp, pc, ip = build_params_from_config(config)

result = run_steady_state(pc, kp, sp, wp, integration_params=ip)

print(result["output"])           # Formatted text report
print(result["converged"])        # True if Newton solver converged
print(result["C"][1, 1])          # Xbh concentration in reactor 1
print(result["FlowWaste"])        # Calculated wastage flow rate
```

### Running a Diurnal Simulation

```python
from uct_activated_sludge.config import (
    load_config, build_params_from_config, load_diurnal_data,
)
from uct_activated_sludge.steady_state import run_steady_state
from uct_activated_sludge.diurnal import run_diurnal

config = load_config("examples/default_plant.toml")
kp, sp, wp, pc, ip = build_params_from_config(config)

# Step 1: Run steady-state for initial conditions
ss = run_steady_state(pc, kp, sp, wp, integration_params=ip)

# Step 2: Load diurnal data
diurnal_data = load_diurnal_data("examples/diurnal_pattern.csv")

# Step 3: Run diurnal simulation
di = run_diurnal(
    plant_config=ss["plant_config"],
    kinetic_params=ss["kinetic_params"],
    stoich_params=sp,
    ww_params=wp,
    integration_params=ip,
    C_steady=ss["CSteady"],
    diurnal_data=diurnal_data,
    stoich_matrix=ss["Stoich"],
)

print(f"Converged: {di['converged']}")
print(f"Cycles: {di['cycle_count']}")
print(f"Data points/day: {di['data_per_day']}")
```

### Inspecting Parameters

```python
from dataclasses import fields
from uct_activated_sludge.models import KineticParams

kp = KineticParams()
for f in fields(kp):
    print(f"{f.name}: {getattr(kp, f.name)}")
```

---

## Output Format

### Steady-State Output

When `--output` is specified, the CLI writes a JSON file with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `C` | float[][] | Concentration matrix `C[k][i]` — compound `i` in reactor `k` (0-indexed in JSON) |
| `C0` | float[] | Influent concentration vector (14 compounds) |
| `CSteady` | float[][] | Steady-state concentrations (used as diurnal initial conditions) |
| `Oc` | float[] | Carbonaceous oxygen uptake rate per reactor (g O₂ m⁻³ h⁻¹) |
| `On` | float[] | Nitrogenous oxygen uptake rate per reactor (g O₂ m⁻³ h⁻¹) |
| `Ot` | float[] | Total oxygen uptake rate per reactor (g O₂ m⁻³ h⁻¹) |
| `Denit` | float[] | Denitrification rate per reactor (g NO₃-N m⁻³ h⁻¹) |
| `converged` | bool | Whether the Newton-Raphson solver converged |
| `FlowWaste` | float | Converged wastage flow rate (ML/d) |

The text report (printed to stdout) shows compound concentrations for all reactors, VSS, TSS, OUR, denitrification rates, and TKN.

### Diurnal Output

When `--output` is specified, the diurnal CLI writes a JSON file with:

| Key | Type | Description |
|-----|------|-------------|
| `cycle_count` | int | Number of 24-hour cycles executed |
| `converged` | bool | Whether periodic steady state was achieved |
| `data_per_day` | int | Number of data points per 24-hour cycle |
| `data_int_hours` | float | Data integration interval in hours |
| `C_final` | float[][] | Final concentration matrix after the last cycle |
| `response` | list | Per-reactor time-series arrays (see below) |

The `response` list is 1-indexed (index 0 is `null`). Each `response[k]` is a 2D array of shape `(19, data_per_day+1)` where rows 1–18 contain time-series data and row 0 is unused padding.

### ASM1 Compound Index

| Index | Symbol | Name | Unit |
|-------|--------|------|------|
| 1 | Xbh | Heterotrophic biomass | g COD m⁻³ |
| 2 | Xe | Endogenous residue | g COD m⁻³ |
| 3 | Xba | Autotrophic biomass | g COD m⁻³ |
| 4 | Xs | Stored COD | g COD m⁻³ |
| 5 | Sbp | Particulate biodegradable COD | g COD m⁻³ |
| 6 | Xi | Particulate unbiodegradable COD | g COD m⁻³ |
| 7 | Xnd | Particulate biodegradable N | g N m⁻³ |
| 8 | Ss | Soluble biodegradable COD | g COD m⁻³ |
| 9 | Snh | Ammonia N | g N m⁻³ |
| 10 | Snd | Soluble organic N | g N m⁻³ |
| 11 | Sno | Nitrate N | g N m⁻³ |
| 12 | Alk | Alkalinity | mole m⁻³ |
| 13 | Si | Soluble unbiodegradable COD | g COD m⁻³ |

Diurnal response arrays include additional derived variables at indices 14–18:

| Index | Symbol | Name | Unit |
|-------|--------|------|------|
| 14 | Oc | Carbonaceous OUR | g O₂ m⁻³ h⁻¹ |
| 15 | On | Nitrogenous OUR | g O₂ m⁻³ h⁻¹ |
| 16 | Ot | Total OUR | g O₂ m⁻³ h⁻¹ |
| 17 | VSS | Volatile suspended solids | g VSS m⁻³ |
| 18 | TKN | Total Kjeldahl nitrogen | g N m⁻³ |

---

## Workflow Examples

### 1. Steady-State with Default Configuration

```bash
uct-asp steady-state --config examples/default_plant.toml
```

This runs a steady-state simulation of a 3-reactor UCT process (1.5 ML anaerobic + 3.0 ML anoxic + 6.0 ML aerobic) with raw sewage at 500 mg/L COD and 50 mg/L TKN, 20°C, 20-day SRT.

### 2. Steady-State with Custom Configuration

Create a custom TOML file (e.g., `my_plant.toml`) modifying only the parameters you want to change:

```toml
[wastewater]
Sti = 600.0
Nti = 60.0

[plant]
Temp = 15.0
Rs = 25.0
```

Then run:

```bash
uct-asp steady-state --config my_plant.toml --output results.json
```

### 3. Diurnal Simulation with Flow Pattern

```bash
uct-asp diurnal \
  --config examples/default_plant.toml \
  --diurnal-data examples/diurnal_pattern.csv \
  --output diurnal_results.json
```

This first runs a steady-state solve, then simulates 24-hour cycles with the realistic flow pattern until convergence.

### 4. Parameter Inspection

View default kinetic parameters:

```bash
uct-asp params --list kinetics
```

View all available groups:

```bash
for group in kinetics stoichiometry wastewater plant integration; do
  echo "=== $group ==="
  uct-asp params --list $group
done
```

### 5. Settled Sewage Simulation

```bash
uct-asp steady-state --config examples/settled_sewage.toml --output settled_results.json
```

Compare with raw sewage results to see the effect of primary sedimentation on biomass concentrations and nutrient removal.

---

## Troubleshooting

**`Error: Configuration file not found`**
The `--config` path does not exist. Use an absolute path or a path relative to the current working directory.

**`Error: PlantConfig.Rs (sludge retention time) must be > 0`**
The TOML config is missing or has an invalid `Rs` value under `[plant]`. SRT must be a positive number.

**`Error: PlantConfig.FlowFeed must be > 0`**
The influent flow rate is zero or negative. Set `FlowFeed` under `[plant]` to a positive value.

**`Error: PlantConfig.VolumeTotal must be > 0`**
No reactor volumes are defined. Set the `Vol` array under `[plant]` with at least one non-zero entry.

**`Error: Unsupported diurnal data format '.xxx'`**
The `--diurnal-data` file must have a `.csv` or `.toml` extension.

**`Error: diurnal_data must have exactly 12 records`**
The diurnal input file must contain exactly 12 rows (one per 2-hour interval).

**Newton solver does not converge**
Try increasing the sludge age (`Rs`), checking that reactor volumes and flows are physically reasonable, or verifying that the DO setpoints match the aeration flags.

**Diurnal simulation does not converge**
Increase `max_cycles` (default 50) in the Python API, or check that the diurnal flow pattern is not too extreme relative to the plant capacity.
