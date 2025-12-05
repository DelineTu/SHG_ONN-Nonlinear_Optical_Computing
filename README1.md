<br/><br/>
**Author: Hua Tu**

This project implements an optical neural network (ONN) based on **second-harmonic generation (SHG)**  via differentiable digital twins and Lagrangian optimization, successfully applied to a VOWEL speech recognition task with 85% accuracy.

This work extends the **Physics-Aware Training (PAT)** framework by introducing a first-principles SHG forward model paired with a differentiable digital-twin backward pass, demonstrating a hardware-in-the-loop photonic computing pipeline that integrates physical nonlinearity with gradient-based learning.

## Overview

### 1. Code

All source code, key datasets, trained models, and performance visualization figures is in  */code*.

### 1. Report

All source code, key datasets, trained models, and performance visualization figures is in  */code*.


1. **Physics Model (SHG_formula)**: Architected and built the `SHG_formula` computational model from the first principles of the χ(2) light-matter interaction.

2. **Digital Twin (SHG_Digital_Model)**: Generated a synthetic dataset using the `SHG_formula` model and trained a digital surrogate, `SHG_Digital_Model`, to emulate the nonlinear optical response. This model is a 5-layer Multi-Layer Perceptron (MLP_Reg).

3. **Hybrid Training (PAT Pipeline)**: Adapted a Physics-Aware Training (PAT) pipeline where the `SHG_formula` model is used for **forward propagation**, and the differentiable `SHG_Digital_Model` is leveraged for **gradient-based backpropagation**.

   A **Lagrangian-based regularization strategy** was introduced to stabilize training and enforce physically meaningful constraints.

4. **Task Validation**: This hybrid framework successfully trained a VOWEL recognition task, achieving **85% classification accuracy**.

5. **Hardware-in-the-loop Integration**: Developed the complete experimental and computational flow, including data acquisition, parsing, normalization/regularization, calibration procedures, and real-time device control (DMD patterns, spectrometer readout), forming a functional **hardware-in-the-loop photonic computing system**.

## Code Usage Instruction

#### System Configuration

Virtual environment setup, file path configurations, and wandb.ai process tracking methods are detailed in the file named **"environment configuration"**. The *debug_gpu_and_memory.ipynb* file  records debugging processes.

The project was successfully run using the following configurations, as detailed in the report on resolving GPU compatibility issues:

- **Python Version**: 3.7.4 or 3.9.19
- **PyTorch Version**: 1.10.0 + CUDA 11.3 (Recommended)
- **Dependencies**: Please install the required packages listed in `requirements.txt`. 

**Note**: ".ipynb" files preserve previously run results for comparison. It is better to create new files for your own experiments.

## Running the Main Experiments

### **Digital Model Training：**

Datasets in `dt_data` are generated via `SHG_Formula`.

After system configuration, run:

- `mean_model_training.py` (mean model)
- `noise_model_training.py` (noise model)
  to train digital twins.

Monitor progress via wandb.ai and check results in `logs/`.

### **SHG_Formula Model Training：**

#### **Step 1: Generate Dataset**

Run the code in `SHG_forward_model+dataset_generation.ipynb`until the following block：

```
#mean digital twin data
Nx = 20000
Nrepeat = 2

%time xlist, specs_list = take_dt_data(Nx, Nrepeat)
np.savez_compressed(f"dt_data/mean_data.npz", specs_list=specs_list, xlist=xlist)

plot_grid(specs_list[0][:,::10], specs_list[1][:,::10]);
```

This generates and saves a new dataset.

#### **Step 2: Train Digital Model**

- Note: make sure to first delete previous results in `logs/`  before training new digital models; otherwise, only the best model will be saved to dt_model.
- Ensure trained models are saved to `dt_models/`.

#### **Step 3: Vowel Recognition Training**

Proceed with the remaining sections of `experiment_SHG_Formula_PAT.ipynb` for task-specific training.

Monitor progress via wandb.ai and check results in `logs/`.

## Citation

If you use this work in your research, please consider citing:

> Hua Tu, "Hardware-in-the-Loop Nonlinear Photonic Computing via Differentiable Digital Twins and Lagrangian Opt

## License and Acknowledgement

Special thanks to the McMahon group at Cornell University for developing the *Physics-Aware Training* (PAT) framework on which this project builds.
- Original Paper:
> Wright, L.G., Onodera, T., Stein, M.M. et al. Deep physical neural networks trained with backpropagation. _Nature_ **601**, 549–555 (2022). https://doi.org/10.1038/s41586-021-04223-6

The code in this repository is released under the following license:

[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

A copy of this license is given in this repository as [license.txt](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/main/license.txt).

