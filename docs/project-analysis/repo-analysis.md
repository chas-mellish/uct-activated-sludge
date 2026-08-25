# Repository Analysis — UCT Activated Sludge Model

## Application Type

DOS-based Turbo Pascal scientific simulation application for modelling biological nutrient removal (BNR) in activated sludge wastewater treatment systems. The program simulates both **steady-state** and **dynamic (diurnal)** behaviour of the UCT Activated Sludge Model, implementing the IWA Activated Sludge Model No. 1 (ASM1) kinetics and stoichiometry.

The original program is a monolithic Turbo Pascal application using the CRT unit for text-mode UI and BGI (Borland Graphics Interface) for graphical output. The Python modernization will strip the UI/graphics layers and expose the simulation engine as a modular Python package.

## Origin and Domain Context

The source code originates from the **University of Cape Town (UCT)** Department of Civil Engineering, developed by **Prof. George Ekama** and colleagues. The UCT Activated Sludge Model is a foundational tool for:

- **Biological Nutrient Removal (BNR)** — nitrogen and phosphorus removal in wastewater
- **Activated sludge process design** — reactor sizing, solids retention time (SRT), food-to-microorganism (F/M) ratio
- **ASM1 implementation** — IAWQ/IWA Activated Sludge Model No. 1 kinetics and stoichiometry
- **Diurnal dynamic simulation** — modelling time-varying influent flow and load patterns
- **Steady-state analysis** — long-term average performance predictions

### Reference Documentation

- **UCT Design Manual** (Chapters 1–6) — design theory and procedures
- **ASM1 papers** — calibration procedures, simulation programs, metabolic profiling, anaerobic digestion

## Repository Structure

uct-activated-sludge/ ├── .gitignore ├── README.md ├── legacy/ │ └── pascal/ # 58 Pascal source files (5,888 lines total) │ ├── ADJTEMP.PAS # Temperature adjustment routines │ ├── BACKGRND.PAS # Background graphics │ ├── BOXES.PAS # UI box drawing │ ├── CFGUNIT.PAS # Configuration unit │ ├── CHANGE.PAS # Parameter change routines │ ├── DATA.PAS # Data handling │ ├── DETECT.PAS # Hardware detection │ ├── DIDATA.PAS # Data I/O │ ├── DIDISK.PAS # Disk I/O │ ├── DIOUTS.PAS # Output routines │ ├── DIURNAL.PAS # Diurnal variation patterns │ ├── DRIVERS.PAS # Device drivers │ ├── DUMPPLOT.PAS # Plot dump routines │ ├── FILEIO.PAS # File I/O operations │ ├── FLOWDI.PAS # Flow diagram routines │ ├── FLOWSS.PAS # Flow calculations │ ├── FONTS.PAS # Font definitions │ ├── FORMAT.PAS # Formatting routines │ ├── FRACINF.PAS # Fraction information │ ├── FRONT.PAS # Front-end routines │ ├── GRAFPLOT.PAS # Graphics plotting │ ├── GRAFUNIT.PAS # Graphics unit │ ├── GROUT.PAS # Graphics output │ ├── INFUNIT.PAS # Information unit │ ├── INITUNIT.PAS # Initialization unit │ ├── INTEGRAT.PAS # Numerical integration │ ├── INTPARAM.PAS # Internal parameters │ ├── IOCHECK.PAS # I/O checking │ ├── IOUNIT.PAS # I/O unit │ ├── KINETIC.PAS # Kinetic parameters │ ├── KINUNIT.PAS # Kinetics unit │ ├── MACHEPS.PAS # Machine epsilon calculation │ ├── MENUIO.PAS # Menu I/O │ ├── MODWASTE.PAS # Waste modification │ ├── NEWTON.PAS # Newton-Raphson solver │ ├── OPUNIT.PAS # Operations unit │ ├── OUTPARAM.PAS # Output parameters │ ├── PAGEUP.PAS # Page update routines │ ├── PRINTER.PAS # Printer output │ ├── PRTUNIT.PAS # Print unit │ ├── RATEUNIT.PAS # Rate calculations unit │ ├── REDIRECT.PAS # Redirect routines │ ├── RETRIEVE.PAS # Data retrieval │ ├── SCALE.PAS # Scaling routines │ ├── SECOND.PAS # Timing routines │ ├── SETAVG.PAS # Set averages │ ├── SETFLG.PAS # Set flags │ ├── SETWASTE.PAS # Set waste parameters │ ├── STCHUNIT.PAS # Stoichiometry unit │ ├── STDYRSLT.PAS # Steady-state results │ ├── STDYUNIT.PAS # Steady-state unit │ ├── STRINGS.PAS # String utilities │ ├── UCTOLD.PAS # Main UCT program │ ├── VARSUNIT.PAS # Global variables unit │ └── WATER.PAS # Water properties ├── docs/ │ ├── ASM1/ # ASM1 reference papers (PDFs, DOCX) │ └── UCT_Design_Manual/ # UCT design theory (Chapters 1-6) ├── src/ # Target: Modernized Python implementation └── converted/ # Target: Morph intermediate files

## Legacy Pascal Source Analysis

**Total:** 58 files, 5,888 lines of Pascal code (Turbo Pascal 4.0–7.0 era)

### File Categorisation by Function

#### Main Program Entry Point

| File | Purpose |
|------|---------|
| `UCTOLD.PAS` | Main program with compiler directives; no numeric coprocessor assumed |

#### Core Computational Modules

| File | Declaration | Purpose |
|------|-------------|---------|
| `VARSUNIT.PAS` | `Unit VarsUnit` | Central global variables and constants |
| `STCHUNIT.PAS` | `Unit StchUnit` | Stoichiometric matrix and coefficients (ASM1) |
| `KINUNIT.PAS` | `Unit KinUnit` | Kinetic parameters and rate expressions |
| `RATEUNIT.PAS` | `Unit RateUnit` | Rate calculation routines |
| `INTEGRAT.PAS` | `Procedure Integrate` | Numerical integration (Dahlquist & Björck method) |
| `NEWTON.PAS` | `Procedure Newton` | Newton-Raphson solver for system of equations |
| `MACHEPS.PAS` | `Procedure CalcMachEps` | Machine epsilon calculation for numerical precision |
| `ADJTEMP.PAS` | `Procedure TempAdjustment` | Temperature correction (Arrhenius) |
| `SCALE.PAS` | `Procedure ScaleValues` | Matrix scaling for numerical stability |
| `FRACINF.PAS` | `Procedure FractionateInfluent` | Influent COD fractionation into ASM1 components |
| `STRINGS.PAS` | `Procedure ComponentNames` | ASM1 component names (Xbh, Xe, Xba, etc.) |

#### Flow and Mass Balance

| File | Declaration | Purpose |
|------|-------------|---------|
| `FLOWDI.PAS` | `Procedure FlowDivision` | Dynamic flow division between reactors |
| `FLOWSS.PAS` | `Procedure FlowDivisionSS` | Steady-state flow division |
| `WASTE.PAS` | `Procedure WastageAndFlows` | Sludge wastage and flow calculations |
| `SETWASTE.PAS` | `Procedure SetWaste` | Initialize wastage pattern from diurnal flow |
| `MODWASTE.PAS` | `Procedure ModifyWastage` | Modify sludge wastage pattern interactively |
| `REDIRECT.PAS` | `Procedure ReDirect` | Flow redirection logic (recycles) |

#### Steady-State Simulation

| File | Declaration | Purpose |
|------|-------------|---------|
| `STDYUNIT.PAS` | `Unit StdyUnit` | Steady-state simulation program unit |
| `STDYRSLT.PAS` | `Procedure OutputSteadyState` | Steady-state result output formatting |
| `SETAVG.PAS` | `Procedure SetAvgInputs` | Set average influent conditions |
| `SETFLG.PAS` | `Procedure SetFlags` | Initialise simulation flags (recycles, etc.) |

#### Dynamic/Diurnal Simulation

| File | Declaration | Purpose |
|------|-------------|---------|
| `DIURNAL.PAS` | `Procedure OutputDiurnalResponse` | Diurnal response output |
| `DIDATA.PAS` | `Procedure WriteARecord` | Write diurnal data records |
| `DIDISK.PAS` | `Procedure StoreOnDisk` | Store diurnal data to disk |
| `DIOUTS.PAS` | `Procedure DiurnalOutputOptions` | Diurnal output menu/options |
| `DIRSLTS.PAS` | `Procedure DiRslts` | Display diurnal results |
| `GRAFUNIT.PAS` | `Unit GrafUnit` | Graphical output of diurnal response |

#### Parameter Management

| File | Declaration | Purpose |
|------|-------------|---------|
| `KINETIC.PAS` | `Procedure Heterotrophs` | Heterotroph kinetic constants |
| `WATER.PAS` | `Procedure WasteWaterUpdate` | Wastewater composition parameters |
| `CHANGE.PAS` | `Procedure ChangeParameter` | Interactive parameter modification |
| `INTPARAM.PAS` | `Procedure CheckIntegParameters` | Integration accuracy and theta parameters |
| `CFGUNIT.PAS` | `Unit CfgUnit` | Configuration management |
| `INFUNIT.PAS` | `Unit InfUnit` | Information unit |
| `OPUNIT.PAS` | `Unit OpUnit` | Operations unit |
| `INITUNIT.PAS` | `Unit InitUnit` | Initialisation routines |
| `DATA.PAS` | `Procedure InitialValues` | Default constant values (menu config, layout) |

#### I/O and File Management

| File | Declaration | Purpose |
|------|-------------|---------|
| `IOUNIT.PAS` | `Unit IOUnit` | General I/O unit |
| `FILEIO.PAS` | `Procedure SetUpFileName` | File naming and extension handling |
| `IOCHECK.PAS` | `Function GetInteger` | Validated integer input |
| `DUMPPLOT.PAS` | `Procedure DumpPlot` | Plot data dump to text file |
| `RETRIEVE.PAS` | `Program Retrieve` | Standalone: read .DID files, convert to text/Lotus format |
| `PAGEUP.PAS` | `Program PageUp` | Standalone: pagination utility |

#### Output and Reporting

| File | Declaration | Purpose |
|------|-------------|---------|
| `PRINTER.PAS` | `Procedure PrintConfiguration` | Process configuration report output |
| `PRTUNIT.PAS` | `Unit PrtUnit` | Print/output unit |
| `OUTPARAM.PAS` | `Procedure SelectedPrintParameters` | Select print output parameters |

#### UI and Graphics (NOT converted — replaced by CLI)

| File | Declaration | Purpose |
|------|-------------|---------|
| `BACKGRND.PAS` | `Procedure ReverseVideo` | Screen colour/video management |
| `BOXES.PAS` | `Procedure InputBox` | Input dialog boxes |
| `DETECT.PAS` | `Procedure DetectScreen` | BGI driver registration (CGA, EGA/VGA) |
| `DRIVERS.PAS` | `Unit Drivers` | BGI graphics drivers (Borland) |
| `FONTS.PAS` | `Unit Fonts` | BGI font linking (Borland) |
| `FORMAT.PAS` | `Procedure Msg` | Screen message formatting |
| `FRONT.PAS` | `Procedure FrontPage` | Title/front page screen |
| `SECOND.PAS` | `Procedure SecondPage` | Second information screen |
| `GRAFPLOT.PAS` | `Procedure PlotGraph` | Graph plotting routine |
| `GROUT.PAS` | `Procedure SelectedGraphParameters` | Graph parameter selection |
| `MENUIO.PAS` | `Procedure FrameInstructions` | Menu frame and instruction borders |

### Key Architectural Observations

1. **Global mutable state:** `VarsUnit` is the central hub — nearly all other units and procedures depend on it for shared variables. This mirrors the Pascal convention of the era and will need careful decomposition in Python.

2. **Include-file pattern:** Many files are not Turbo Pascal `unit`s with `interface`/`implementation` sections, but rather bare procedure files likely `{$I}`-included by the main program (`UCTOLD.PAS`).

3. **Two standalone programs:** `RETRIEVE.PAS` and `PAGEUP.PAS` are standalone `Program` files, not included by the main program.

4. **Borland-provided files:** `DRIVERS.PAS` and `FONTS.PAS` are Borland sample units for BGI graphics linking — these can be excluded from conversion.

5. **Numerical methods:** The integration routine references Dahlquist & Björck, suggesting a Runge-Kutta or similar ODE solver. The `Newton` procedure handles nonlinear system solving.

6. **ASM1 compliance:** `STRINGS.PAS` confirms ASM1 component naming (Xbh = heterotrophic biomass, Xe = endogenous residue, Xba = autotrophic biomass).

7. **Dual simulation modes:** The codebase clearly separates steady-state (`STDY*` files) and dynamic/diurnal (`DI*` files) simulation paths.

## Conversion Strategy Notes

### Files to Convert (Computational Core)

Approximately **35–38 files** contain computational logic that needs conversion:

- All `*UNIT.PAS` units (VarsUnit, StchUnit, KinUnit, RateUnit, etc.)
- Computational procedures (Integrate, Newton, FlowDivision, FractionateInfluent, etc.)
- I/O and file management (adapt to Python file operations)

### Files to Exclude (UI/Graphics Only)

Approximately **12–15 files** are pure UI/graphics with no computational value:

- `BACKGRND.PAS`, `BOXES.PAS`, `DETECT.PAS`, `DRIVERS.PAS`, `FONTS.PAS`
- `FORMAT.PAS`, `FRONT.PAS`, `SECOND.PAS`, `GRAFPLOT.PAS`, `GROUT.PAS`
- `MENUIO.PAS`
- Parts of `DATA.PAS` (menu layout config), `CHANGE.PAS` (interactive prompts)

### Proposed Python Module Mapping

src/uct_activated_sludge/ ├── init.py # Package exports ├── models.py # Dataclasses replacing VarsUnit globals ├── constants.py # ASM1 stoichiometric/kinetic constants ├── stoichiometry.py <- StchUnit.PAS ├── kinetics.py <- KinUnit.PAS, KINETIC.PAS, RATEUNIT.PAS ├── influent.py <- FRACINF.PAS, WATER.PAS ├── hydraulics.py <- FLOWDI.PAS, FLOWSS.PAS, WASTE.PAS, SETWASTE.PAS ├── solver.py <- NEWTON.PAS, MACHEPS.PAS ├── integrator.py <- INTEGRAT.PAS, INTPARAM.PAS ├── temperature.py <- ADJTEMP.PAS ├── steady_state.py <- STDYUNIT.PAS, STDYRSLT.PAS, SETAVG.PAS, SETFLG.PAS ├── diurnal.py <- DIURNAL.PAS, DIDATA.PAS, DIOUTS.PAS, DIRSLTS.PAS ├── io/ <- FILEIO.PAS, IOCHECK.PAS, IOUNIT.PAS, DIDISK.PAS │ ├── init.py │ ├── file_io.py │ └── validation.py ├── output/ <- PRINTER.PAS, PRTUNIT.PAS, OUTPARAM.PAS │ ├── init.py │ └── reports.py └── cli.py # New: replaces all UI/menu procedures

## Acknowledgement

Original UCT Activated Sludge Model code developed by Prof. George Ekama and colleagues at UCT.
Available for educational and research purposes.