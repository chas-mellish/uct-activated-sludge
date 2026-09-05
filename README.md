# UCT Activated Sludge Model

A Python 3.10+ implementation of the UCT (University of Cape Town) Activated Sludge Model, based on the IWA Activated Sludge Model No. 1 (ASM1) for Biological Nutrient Removal (BNR) simulation. Converted from the original Turbo Pascal 7.0 code developed by Prof. George Ekama at UCT.

The package simulates steady-state and dynamic (diurnal) behaviour of multi-reactor activated sludge systems, solving for all 13 ASM1 state variables using Newton-Raphson iteration and adaptive predictor-corrector ODE integration.

## Quick Start

Install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

Run a steady-state simulation with the default 3-reactor UCT plant:

```bash
uct-asp steady-state --config examples/default_plant.toml
```

Save results as JSON:

```bash
uct-asp steady-state --config examples/default_plant.toml --output results.json
```

Run a diurnal simulation with a realistic flow pattern:

```bash
uct-asp diurnal \
  --config examples/default_plant.toml \
  --diurnal-data examples/diurnal_pattern.csv \
  --output diurnal_results.json
```

Inspect default parameter values:

```bash
uct-asp params --list kinetics
```

For full documentation, see the [Usage Guide](docs/usage-guide.md).

## Project Structure

```
src/uct_activated_sludge/   Python package (14 modules)
  cli.py                    CLI entry point (uct-asp command)
  config.py                 TOML configuration loader
  models.py                 Typed dataclasses (parameter groups)
  stoichiometry.py           Stoichiometric matrix builder
  kinetics.py               Process rates and air supply
  influent.py               COD/TKN fractionation
  hydraulics.py             Flow division and wastage
  solver.py                 Newton-Raphson solver
  integrator.py             Predictor-corrector ODE integrator
  temperature.py            Arrhenius temperature correction
  steady_state.py           Steady-state simulation driver
  diurnal.py                Dynamic diurnal simulation
  constants.py              ASM1 enums and sizing constants

examples/                   Example configuration files
  default_plant.toml        Raw sewage, 3-reactor UCT process
  settled_sewage.toml       Settled sewage fractionation
  diurnal_pattern.csv       Realistic 12-record diurnal flow pattern

tests/                      Test suite (pytest + Hypothesis)
legacy/pascal/              Original Pascal source (58 .PAS files)
docs/                       Documentation and reference materials
  usage-guide.md            Comprehensive usage reference
```

## Requirements

- Python >= 3.10
- NumPy >= 1.26.0
- SciPy >= 1.12.0
- pandas >= 2.1.0

Dev dependencies: pytest, pytest-cov, Hypothesis, matplotlib.

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

See below for acknowledgement of the original UCT source material.

## Acknowledgement

This project contains legacy Pascal code and reference documentation originating from the University of Cape Town (UCT), originally developed by Professors G vR Marais, George Ekama, Richard E Lowenthal and colleagues.

The original materials are provided for educational and research purposes. Please acknowledge UCT and Prof. Ekama's contributions when using this code.

For questions about licensing or commercial use, contact the UCT Department of Civil Engineering.
