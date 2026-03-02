# NOSTRUM Blade Surrogate Model Training

MPI-enabled C++ workflow to train a surrogate model from a higher fidelity computation of the output function. 
In the context of the NOSTRUM project the shared code is implemented for sample blade aerodynamic optimization using AeroDyn/OpenFAST (https://github.com/OpenFAST/openfast) as high-fidelity evaluation tool.
The high fidelity module is defined as a separate Class that can be customized by the developer, changing the external solver to higher (CFD, digital twin, experimental facility, ...) or lower fidelity ones.

## Requirements

- C++20 compiler
- CMake >= 3.8
- MPI implementation (`mpirun`, MPI C/C++ compilers)
- BLAS + LAPACK
- `aerodyn_driver` (v3.5.0) executable from OpenFAST (path configured in `config_HF_Model.txt`)

## Build / Installation

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Executable:

- `build/TrainSurrogateModel`

## Configuration

Two config files are used:

### `config_main.txt`

Main training settings:

- `project_path` : absolute project root path
- `pointsPerParam` : points per parameter for prediction (uniform) grid generation
- `nk` : number of RBF shape-parameter samples used by SRBF uncertainty
- `iterMax` : max adaptive iterations
- `tolerance` : stopping tolerance on SRBF variance

Example:

```txt
project_path=/absolute/path/to/NOSTRUM-BladeOptimization/
pointsPerParam=6;
nk=100;
iterMax=200;
tolerance=0.001;
```

### `config_HF_Model.txt`
NOT: This file's structure and variables are consistent with the implementation that assumes the use of the AeroDyn setup provided in this version. In case of changing of the HF model, the user should also update the input and configuration sections.
High-fidelity model and design-variable settings:

- `blade_file_name`
- `aerodyn_path`
- `twistID`, `chordID`, `spanID`
- `torsionStartID`
- `rotCpChannel`
- `CP_indices` (count on first line, comma-separated values on next line)
- `UB` (count + values on next line)
- `LB` (count + values on next line)

Example:

```txt
blade_file_name=BLADE_FILE.dat
aerodyn_path=/path/to/openfast/install/bin/aerodyn_driver
twistID=5;
chordID=6;
spanID=1;
torsionStartID=15;
rotCpChannel=26;
CP_indices=5
12, 21, 30, 39, 49;
UB=6
10.0, 5.0, 4.0, 3.0, 2.0, 3.0;
LB=6
4.0, -1.0, -2.0, -3.0, -2.0, 0.0;
```

## Usage

Run with default config (`config_main.txt`):

```bash
mpirun -np 4 ./build/TrainSurrogateModel
```

Run with explicit config path:

```bash
mpirun -np 4 ./build/TrainSurrogateModel /absolute/or/relative/path/to/config_main.txt
```

Serial run (for comparison/debug):

```bash
mpirun -np 1 ./build/TrainSurrogateModel
```

## Outputs

- `Results/SM_training/SM_Database.txt` : surrogate database (training points + responses)
- `Results/SM_training/HF_Model_info.txt` : stored HF reference/control-point data
- `train_log.dat` : appended training history
- `LOG_Files/log.Aerodyn1`, `LOG_Files/log.Aerodyn2` : AeroDyn run logs

## Code Structure

- `src/SurrogateModelTraining_NOSTRUM_Blade.cpp`  
  Main MPI driver: load configs, initialize/broadcast parameters, adaptive sampling loop, SRBF max-variance point selection, HF evaluation on rank 0, and synchronization.
- `src/HF_Model.cpp` + `src/compute_cp_w_torsion_dof.cpp`  
  HF model setup and AeroDyn-based `computeFT`.
- `src/compute_srbf.cpp` + `src/rbf.cpp`  
  SRBF uncertainty calculation and RBF solve (LAPACK-backed linear solve).
- `src/generate_param_grid.cpp` + `src/generate_combinations.cpp`  
  Cartesian prediction-grid generation from parameter bounds.
- `include/`  
  Public interfaces for all modules.
- `AeroDynFiles/`  
  AeroDyn driver files, airfoil data, and blade input files used by HF evaluations.

# NOSTRUM Project summary 

NOSTRUM is a 2-year research project aimed at optimizing floating offshore wind turbines (FOWTs) for the Mediterranean Sea. Through a combination of advanced numerical approaches, the project focuses on innovative design solutions for blades, towers and floating platforms, along with advanced control strategies tailored to Mediterranean conditions.

The aim is to provide the wind energy community of design guidelines and reference data, models and analyses to apply in the framework of modern floating wind turbines to use in the target framework of offshore wind energy development and research in our Sea.

This collaborative effort between three research units from Sapienza University, Università degli Studi di Firenze, and CNR-INM will culminate in a new reference wind turbine model, available as an open-access resource for future research and feasibility studies.

Visit at: https://nostrum-project.it

# References 

[1] Cardamone, R., Broglia, R., Papi, F., Rispoli, F., Corsini, A., Bianchini, A., & Castorrini, A. (2025). Aerodynamic design of wind turbine blades using multi-fidelity analysis and surrogate models. International Journal of Turbomachinery, Propulsion and Power, 10(3), 16. https://doi.org/10.3390/ijtpp10030016

[2] Castorrini, A., Papi, F., Bianchini, A., Broglia, R., Rispoli, F., Ferrara, G., Lugni, C., & Cardamone, R. (2026). NOSTRUM - Optimizing Floating Offshore Wind Turbines For Use In The Mediterranean Sea - TECHNICAL REPORTS - Part 2. Zenodo. https://doi.org/10.5281/zenodo.18620593

# Acknowledgement

This research is supported by the Ministry of University and Research (MUR) as part of the European Union program NextGenerationEU, Missione 4 “Istruzione e Ricerca” del Piano Nazionale di Ripresa e Resilienza (PNRR), Componente C2 – Investimento 1.1, Fondo per il Programma Nazionale di Ricerca e Progetti di Rilevante Interesse Nazionale (PRIN).
