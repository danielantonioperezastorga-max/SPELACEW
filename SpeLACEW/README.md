# SpelacEW

Interactive spectral line fitting and Equivalent Width (EW) measurement tool for stellar spectroscopy.

SpelacEW is a Python-based interactive software designed for the analysis of stellar absorption lines through manual continuum placement and Gaussian profile fitting. The code allows the user to inspect spectral regions individually, perform single-line or multi-line (blended) fits, and export reproducible measurements in a structured format.

The project was developed for high-control spectroscopic analysis workflows where visual inspection and user-guided continuum selection are essential.

---

# Features

- Interactive visualization of stellar spectra
- Support for:
  - FITS spectra
  - ASCII/TXT/CSV spectra
- Manual continuum placement
- Gaussian fitting of absorption lines
- Multi-Gaussian fitting for blended lines
- Automatic calculation of:
  - Equivalent Width (EW)
  - Full Width at Half Maximum (FWHM)
  - Reduced chi-square (χ²)
- Reference continuum visualization from external datasets
- Interactive spectral exploration mode
- Automatic generation of:
  - CSV results tables
  - Multi-page PDF reports
- Session accumulation:
  - Existing results are preserved between runs
  - PDFs are automatically merged across sessions

---

# Scientific Motivation

Precise measurement of spectral line parameters is a fundamental task in stellar spectroscopy and abundance analysis. While fully automated EW pipelines are efficient for large datasets, many scientific applications still require manual inspection due to:

- blended spectral features
- continuum uncertainties
- local noise structures
- line asymmetries
- crowded spectral regions

SpelacEW was designed as a semi-manual high-control alternative, prioritizing:

- reproducibility
- visual inspection
- interactive continuum definition
- robust treatment of blended lines

This makes the code particularly useful for:

- stellar abundance studies
- chemical tagging
- solar reference comparisons
- educational spectroscopy workflows
- spectroscopic quality control

---

# Installation

## Clone repository

```bash
git clone https://github.com/USERNAME/spelacew.git
cd spelacew
```

## Install package

Recommended installation:

```bash
pip install -e .
```

This enables the command-line interface:

```bash
spelacew
```

---

# Dependencies

Main dependencies:

- numpy
- scipy
- matplotlib
- pandas
- PyAstronomy
- pypdf

Python >= 3.9 is required.

---

# Input Data

## Spectrum

Supported formats:

- `.fits`
- `.fit`
- `.txt`
- `.dat`
- `.ascii`
- `.csv`

ASCII files must contain:

```text
wavelength flux
```

in two columns.

---

## Line List

The line list must be provided as a CSV file.

Minimum required column:

```text
wavelength, element, species, ep, gf
```

Example:

```csv
wavelength,element,species,ep,gf
6562.79,H,1,10.2,-0.3
```

---

# Usage

## Interactive mode

Run:

```bash
spelacew
```

The program will ask interactively for:

- spectrum file
- line list
- optional reference file
- optional width

---

## Command-line mode

```bash
spelacew spectrum.fits lines.csv
```

Optional arguments:

```bash
spelacew spectrum.fits lines.csv width
```

or

```bash
spelacew spectrum.fits lines.csv width reference.csv
```

Example:

```bash
spelacew HD10700.fits lines_fe.csv 1.5 solar_reference.csv
```

---

# Python / IPython Usage

SpelacEW can also be executed directly from Python or IPython environments.

This is useful for:

- interactive scientific workflows
- Jupyter notebooks
- debugging
- custom spectroscopy pipelines
- integration with other analysis tools

---

## Importing the package

```python
from spelacew import EW
```

---

## Interactive execution

Run the interactive interface:

```python
EW.run()
```

The program will request:

- spectrum file
- line list
- optional reference EW file
- optional visualization width

---

## Example

```python
from spelacew import EW

EW.run()
```

---

## Direct object creation

You may also instantiate the class manually:

```python
from spelacew import EW

obj = EW(
    "HD10700.fits",
    "lines.csv",
    "solar_reference.csv",
    1.5
)
```

Parameters:

| Parameter | Description |
|---|---|
| spectrum file | Observed spectrum |
| line list | CSV containing spectral lines |
| star reference | Optional reference EW dataset |
| width | Spectral visualization window in Å |

---

## Recommended IPython Workflow

Example:

```bash
ipython
```

```python
from spelacew import EW

EW.run()
```

This mode is particularly convenient during:

- interactive spectrum inspection
- spectroscopic teaching activities
- exploratory stellar spectroscopy

# Interactive Interface

The interface displays:

- observed spectrum
- theoretical line center
- crosshair cursor
- local spectral window
- automatic zoom panel after fitting
- reference continuum visualization (optional)

The user interacts entirely through keyboard commands.

---

# Controls

## Navigation

| Key | Action |
|---|---|
| `n` | Next spectral line |
| `p` | Previous spectral line |

---

## Normal Fitting Mode

| Key | Action |
|---|---|
| `k` | Select continuum points |

Procedure:

1. Press `k` on left continuum region
2. Press `k` on right continuum region
3. Fit executes automatically

The continuum is estimated through a linear interpolation between the selected points.

---

## Blending Mode

| Key | Action |
|---|---|
| `b` | Enable/disable blending mode |
| `d` | Define fitting region |
| `g` | Add Gaussian component center |
| `Enter` | Execute multi-Gaussian fit |

This mode is intended for partially or strongly blended spectral features.

Each Gaussian component is fitted simultaneously.

---

## Exploration Mode

| Key | Action |
|---|---|
| `x` | Enable exploration mode |
| `c` | Manual zoom input |
| `w` | Change visualization width |

Exploration mode allows free navigation through the spectrum independently of the predefined line list.

---

## Other Commands

| Key | Action |
|---|---|
| `r` | Full reset |
| `a` | Cancel input |
| `q` | Save and quit |

---

# Outputs

For each session, the program generates a directory:

```text
ajustes_<spectrum_name>_<date>/
```

Example:

```text
ajustes_HD10700_2026-05-06/
```

---

## CSV Results

File:

```text
resultados.csv
```

Contains:

| Column | Description |
|---|---|
| wavelength | Central wavelength |
| wave_left | Left fitting boundary |
| wave_right | Right fitting boundary |
| left_continuum | Left continuum level |
| right_continuum | Right continuum level |
| element | Atomic species |
| species | Ionization state |
| ep | Excitation potential |
| gf | Oscillator strength |
| ew_ref | Reference EW |
| ew | Measured EW |
| FWHM | Full Width at Half Maximum |
| Chi2R | Reduced χ² |
| hpf | Reserved parameter |

Equivalent Width values are reported in milli-Angstroms (mÅ).

---

## PDF Report

File:

```text
resultados.pdf
```

Contains one page per fitted line including:

- observed spectrum
- continuum model
- Gaussian fit
- zoom visualization
- fit diagnostics

PDF sessions are automatically merged between runs.

---

# Fitting Methodology

## Continuum Normalization

The continuum is defined manually using two user-selected points.

A linear continuum model is constructed through:

```math
F_c(\lambda) = a\lambda + b
```

The observed spectrum is normalized as:

```math
F_{norm} = \frac{F}{F_c}
```

---

## Gaussian Model

Single-line fits use:

```math
F(\lambda) =
1 - A \exp
\left(
-\frac{(\lambda-\mu)^2}{2\sigma^2}
\right)
```

where:

- \(A\) = line depth
- \(\mu\) = line center
- \(\sigma\) = Gaussian width

---

## Multi-Gaussian Blending

For blended lines:

```math
F(\lambda)=
1-
\sum_i
A_i
\exp
\left(
-\frac{(\lambda-\mu_i)^2}{2\sigma_i^2}
\right)
```

Each component is fitted simultaneously.

---

## Equivalent Width

Equivalent Width is computed numerically as:

```math
EW =
\int
(1-F_{norm}) d\lambda
```

and reported in milli-Angstroms (mÅ).

---

# Recommended Usage

For more stable fits:

- select continuum regions wider than the line core
- avoid noisy edge regions
- inspect blended features manually
- verify continuum placement visually

Broader continuum selections generally improve normalization stability and reduce sensitivity to local noise fluctuations.

---

# Current Status

Current implemented features:

- Interactive continuum selection
- Single Gaussian fitting
- Multi-Gaussian blending
- EW/FWHM/χ² calculations
- PDF and CSV export
- Reference EW comparison
- Spectral exploration mode
- Session accumulation

Planned future developments:

- automatic continuum estimation
- Voigt profile support
- radial velocity correction tools
- abundance pipeline integration
- GUI improvements
- batch-processing utilities

---

# Citation

If you use SpelacEW in scientific work, please cite the repository and acknowledge the software appropriately.

---

# Author

Daniel Pérez  
Universidad de Concepción

---
