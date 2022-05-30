# PINNs-Experiments

Using physics-informed neural networks (PINNs) for solving atomic time-dependent Schrödinger equation (TDSE) in perturbative regime. Various experiments are available within the project framework:
* Hydrogen orbitals (solving unperturbed atomic time-independent Schrödinger equation (TISE))
* Hydrogen atom in a DC Stark setting
* Unperturbed hydrogen orbital evolution (solving unperturbed atomic TDSE)
* One-level atom in an AC Stark setting

## Setup

Create and activate the `conda` environment:

```console
conda env create -f conda.yml
conda activate pinns-experiments
```

## Styling

Enforce styling with `black`:

```console
python -m black src
```

## Getting Started


