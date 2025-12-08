<br/><br/>
# A dual-digital-twin nonlinear optical neural network enabling hardware-in-the-loop, gradient-based learning through χ² dynamics.

This repository contains a hardware-in-the-loop (HIL), end-to-end trainable nonlinear optical neural network (ONN) based on second-harmonic generation (SHG).
It demonstrates a full photonic computing pipeline that couples:

- **Dual differentiable digital-twin modeling**:
  - An experiment-calibrated twin trained directly on measured SHG I/O data, capturing real-system nonlinear response, saturation, drift, and stochastic fluctuations.
  - A first-principles-based twin trained on synthetic data generated from SHG physical model, providing an interpretable baseline for validating model convergence and algorithm behavior.
- **Physics-aware training (PAT)**: combines SHG forward propagation with noise-aware learned backward digital twins.
- **Lagrangian-constrained optimization**: ensures safe voltage/optical-power actuation during training.
- **End-to-end HIL architecture**: implements a hardware-ready PyTorch layer that encodes the input spectrum, organizes data transfer to and from the SHG instrument stack, and integrates custom regularization and calibration routines for stable training and hardware-faithful execution.
- **Twin comparison**: I trained the SHG-ONN on the VOWEL task using two digital twins:
  - Experiment-calibrated twin (fits real SHG data)
  - Physics-model twin (fits synthetic χ² model data)
- **Findings**:
  - The experiment-calibrated twin yields higher downstream task accuracy and better stability, because it embeds real-system distortions and noise.
  - The physics-model twin trains faster but yields lower accuracy in final performance, reflecting the mismatch between ideal physics and physical hardware.

This SHG-ONN was successfully applied to a VOWEL speech-recognition task, achieving **~85% test accuracy**, validating the scalability, robustness, and trainability of nonlinear optical processors.

## 1. Main Results

### (A) VOWEL Classification - Training Curves
<img width="1570" height="526" alt="image" src="https://github.com/user-attachments/assets/d9172802-8ce9-41be-be84-1571dae77fb8" />

### (B) Physics-Aware Training Framework
<img width="855" height="396" alt="image" src="https://github.com/user-attachments/assets/09028603-99b9-4e06-adb4-37a8d56e7354" />

### (C) Digital-Twin Learning Curves
<img width="2365" height="1274" alt="image" src="https://github.com/user-attachments/assets/68403b72-dde8-430b-a207-b3389b2f4d71" />

### (D) SHG Digital Twin Fitting Results
<img src="https://cdn.jsdelivr.net/gh/DelineTu/Pictures/202501262100860.png" alt="image-20250126203544673" style="zoom:33%;" />



## 2. Repository Overview

### 1. Code

All source code, key datasets, trained models, and performance visualization figures is in  */code*.

### 1. Report

1. **Physics Model (SHG_formula)**: Architected and built the `SHG_formula` computational model from the first principles of the χ(2) light-matter interaction.

2. **Digital Twin (SHG_Digital_Model)**: Generated a synthetic dataset using the `SHG_formula` model and trained a digital surrogate, `SHG_Digital_Model`, to emulate the nonlinear optical response. This model is a 5-layer Multi-Layer Perceptron (MLP_Reg).

3. **Hybrid Training (PAT Pipeline)**: Adapted a Physics-Aware Training (PAT) pipeline where the `SHG_formula` model is used for **forward propagation**, and the differentiable `SHG_Digital_Model` is leveraged for **gradient-based backpropagation**.

4. **Lagrangian-based regularization strategy** was introduced to stabilize training and enforce physically meaningful constraints.

4. **Task Validation**: This hybrid framework successfully trained a VOWEL recognition task, achieving **85% classification accuracy**.

5. **Hardware-in-the-loop Integration**: Developed the complete experimental and computational flow, including data acquisition, parsing, normalization/regularization, calibration procedures, and real-time device control (DMD patterns, spectrometer readout), forming a functional **hardware-in-the-loop photonic computing system**.

## Code Usage Instruction

## System Configuration

Virtual environment setup, file path configurations, and wandb.ai process tracking methods are detailed in the file named **"environment configuration"**. The *debug_gpu_and_memory.ipynb* file documents the debugging process.

The project was run using the following configurations:

- **GPU**: RTX 4060
- **Python Version**: 3.7.4
- **PyTorch Version**: 1.10.0 + CUDA 11.3
- **Dependencies**: Please install the required packages listed in `requirements_pip.txt`(pip install) and  `requirements_conda.txt`(conda install).
  
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

This project includes modified material from the *Physics-Aware Training* (PAT) framework by the McMahon group at Cornell University (CC BY 4.0 license). I gratefully acknowledge the McMahon Group for developing the differentiable physical computing framework on which this project builds.

Reference:
> Wright, L.G., Onodera, T., Stein, M.M. et al. Deep physical neural networks trained with backpropagation. _Nature_ **601**, 549–555 (2022). https://doi.org/10.1038/s41586-021-04223-6

The code in this repository is released under the following license:

[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

A copy of this license is given in this repository as license.txt.







