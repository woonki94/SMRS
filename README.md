# SMRS

## Sparse Modeling for Finding Representative Objects (SMRS) Algorithm

This project implements the **Sparse Modeling for Finding Representative Objects (SMRS)** algorithm using the **Alternating Direction Method of Multipliers (ADMM)** optimization approach.

The SMRS algorithm was introduced in the paper **"See All by Looking at A Few: Sparse Modeling for Finding Representative Objects"** by *Ehsan Elhamifar, Guillermo Sapiro, and René Vidal*. [Link to paper](https://ieeexplore.ieee.org/document/6247852).

## Overview

This repository provides:
- An analysis of the method presented in the paper.
- A Python implementation of the SMRS algorithm with my interpretation.

### Key Features of SMRS:
- **Representative Selection**: Efficiently selects a subset of representative objects from a dataset.
- **Sparse Representation**: Utilizes sparse modeling to reconstruct the dataset using a few key representatives.
- **Outlier Detection**: Identifies and excludes non-representative or outlier data points.
- **Scalable Optimization**: Employs ADMM for convex relaxation and efficient optimization.

### Experiments

#### Synthetic Experiments
1. **Separable Convex Hulls**: 
   - Tested the algorithm on a synthetic dataset with two separable convex hulls to evaluate the selection of representatives and their placement relative to the dataset's geometry.
   
2. **Overlapping Convex Hulls**: 
   - Evaluated the performance on a dataset with two overlapping convex hulls, analyzing how well the algorithm handles complex data distributions and selects representatives.

#### Practical Experiments
- **MNIST Fashion Dataset**:
   - Applied SMRS to select representative data points from the MNIST Fashion dataset.
   - Used the selected representatives for classification tasks and observed their effectiveness in maintaining classification accuracy with reduced data.


## References
1. **Paper**: Ehsan Elhamifar, Guillermo Sapiro, and René Vidal, ["See All by Looking at A Few: Sparse Modeling for Finding Representative Objects"](https://ieeexplore.ieee.org/document/6247852), CVPR 2012.
