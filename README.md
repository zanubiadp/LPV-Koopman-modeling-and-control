## Repository contents

This repository contains both `.py` scripts and `.ipynb` notebooks.

When a Python script and a notebook share the same name, they implement the same functionality. The notebooks are mainly intended for testing, debugging, and exploratory analysis, since they allow you to focus on specific parts of the pipeline without rerunning the entire codebase.

If you notice small mismatches between a `.py` file and its corresponding `.ipynb`, they are temporary and will be aligned over time.

All other files are mainly supporting functions that are used in the main scripts and notebooks.

## How to use this repository

The main scripts are:

- `modelAnalysis.py`  
  Used to study and visualize the system dynamics. This step helps verify that the system is a meaningful candidate for the Koopman learning process before running the full pipeline.

- `main.py` or `main.ipynb`  
  This is the core script where the actual learning takes place, from eigenvalue estimation to eigenfunction learning.

- `replay_plots.py` or `replay_plots.ipynb`  
  Useful for post-analysis. After training, these scripts allow you to revisit and analyze behaviors that were not explicitly explored in `main`. Since interpolants cannot be saved in a file, eigenfunctions must be recomputed from their saved sample points.
