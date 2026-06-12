mpes-analysis 

Python tools for multidimensional photoemission spectroscopy data analysis, visualization, and interactive exploration.

This repository contains:
- Jupyter notebooks for reproducible data processing workflows and figure generation
- Reusable and standardized Python modules for loading ARPES data, plotting, curve fitting, etc.
- Interactive GUI for data exploration of multidimensional datasets.

Key Features:
- Momentum map extraction
- Energy-momentum or energy-delay cuts
- Delay trace analysis and curve fitting
- FFT-based analysis: Real-space reconstruction from momentum map signatures
- Use of simple and intuitive reusable functions for plotting or feature extraction
- Interactive GUI
- Use of xarray data structures

Structure:
mpes-analysis/
├── notebooks/      # Notebooks: analysis and workflows
├── arpes_tools/    # Data loading, analysis, and plotting functions
├── gui/            # Interactive GUI
└── README.md

Dependencies
NumPy
SciPy
Matplotlib
h5py
Jupyter
xarray

Lawson T. Lloyd
FHI Berlin
2026