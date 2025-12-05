# SHG-ONN_Nonlinear-Optical-Computing
**Author: Hua Tu**

This project implements an optical neural network (ONN) based on **second-harmonic generation (SHG)**  via differentiable digital twins and Lagrangian optimization, successfully applied to a VOWEL speech recognition task with 85% accuracy.

This work extends the **Physics-Aware Training (PAT)** framework by introducing a first-principles SHG forward model paired with a differentiable digital-twin backward pass, demonstrating a hardware-in-the-loop photonic computing pipeline that integrates physical nonlinearity with gradient-based learning.

#### 1. Key Achievements and Contributions

------

1. **Physics Model (SHG_formula)**: Architected and built the `SHG_formula` computational model from the first principles of the χ(2) light-matter interaction.

2. **Digital Twin (SHG_Digital_Model)**: Generated a synthetic dataset using the `SHG_formula` model and trained a digital surrogate, `SHG_Digital_Model`, to emulate the nonlinear optical response. This model is a 5-layer Multi-Layer Perceptron (MLP_Reg).

3. **Hybrid Training (PAT Pipeline)**: Adapted a Physics-Aware Training (PAT) pipeline where the `SHG_formula` model is used for **forward propagation**, and the differentiable `SHG_Digital_Model` is leveraged for **gradient-based backpropagation**.

   A **Lagrangian-based regularization strategy** was introduced to stabilize training and enforce physically meaningful constraints.

4. **Task Validation**: This hybrid framework successfully trained a VOWEL recognition task, achieving **85% classification accuracy**.

5. **Hardware-in-the-loop Integration**: Developed the complete experimental and computational flow, including data acquisition, parsing, normalization/regularization, calibration procedures, and real-time device control (DMD patterns, spectrometer readout), forming a functional **hardware-in-the-loop photonic computing system**.

#### 2. Overview

------

Code:

All source code, key datasets, trained models, and performance visualization figures is in  `/code`

This project builds a hybrid optical-computational pipeline that integrates:

- 
- **Instrument control** (DMD + spectrometer + GPIB devices)
- **SHG-based nonlinear optical transformation**
- **Differentiable digital twins for backpropagation**
- **Task-level training on VOWEL classification**

Report

The full technical report is in: `/report/SHG_ONN_Research_Report.pdf`

It includes:

- System architecture
- Physics-informed SHG model derivation
- Digital twin construction
- PAT framework
- VOWEL recognition results
- Discussion of stability, noise, and optimization strategies

#### 1. Getting Started

------

#### System Configuration

Virtual environment setup, file path configurations, and wandb.ai process tracking methods are detailed in the file named **"environment configuration"**. The `debug_gpu_and_memory.ipynb` file  records debugging processes.

The project was successfully run using the following configurations, as detailed in the report on resolving GPU compatibility issues:

- **Python Version**: 3.7.4 or 3.9.19
- **PyTorch Version**: 1.10.0 + CUDA 11.3 (Recommended)
- **Dependencies**: Please install the required packages listed in `requirements.txt`. 

**Note**: ".ipynb" files preserve previously run results for comparison. It is better to create new files for your own experiments.

#### Running the Main Experiments

##### **Digital Model Training：**

Datasets in `dt_data` are generated via `SHG_Formula`.

After system configuration, run:

- `mean_model_training.py` (mean model)
- `noise_model_training.py` (noise model)
  to train digital twins.

Monitor progress via wandb.ai and check results in `logs/`.

##### **SHG_Formula Model Training：**

###### **Step 1: Generate Dataset**

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

###### **Step 2: Train Digital Model**

- Note: make sure to first delete previous results in `logs/`  before training new digital models; otherwise, only the best model will be saved to dt_model.
- Ensure trained models are saved to `dt_models/`.

###### **Step 3: Vowel Recognition Training**

Proceed with the remaining sections of `experiment_SHG_Formula_PAT.ipynb` for task-specific training.

Monitor progress via wandb.ai and check results in `logs/`.

#### 2. Technical Reports Overview

------

Decomposed the project into physically interpretable operations and training pipeline modules for clear understanding.

Main contents of the html. files:

1. Optical Instrument Control, Data Parsing, and Testing
2. ==Development and Training of the *SHG_formula* Model==
3. Wavelength Parameter Configuration
4. Acquisition of Spectral Outputs from Sample Sets
5. Collection of the SHG_Digital_Model Training Dataset
   *(Mathematical model training methodology was covered in the previous report.)*
6. Training Preparation:
   (1) Encapsulate the 100→50 dimensional physical transformation as a PyTorch-compatible function `f_exp`.
   (2) Load the mathematical model and construct forward/backward propagation functions `f_pat`.
7. Training Preparation: Retrain the pretrained Digital Model for the current experiment.
8. Deep Learning Model Construction:
   (1) Loss function definition
   (2) Build a single-layer forward propagation module (with 50+50 learnable parameters) based on PyTorch instance `f_exp`
   (3) Assemble the full deep learning model (5 layers)
   (4) Define model evaluation metrics
9. Training and Model Evaluation





### Citation

If you reference this work, please cite:

```Python#
Hua Tu, "Hardware-in-the-Loop Nonlinear Photonic Computing via Differentiable Digital Twins and Lagrangian Opt
```



#### License and Attribution

This project builds upon the **Physics-Aware Training (PAT)** framework released by the McMahon group at Cornell University.

- **Original Code Repository:** https://github.com/mcmahon-lab/Physics-Aware-Training

- **Original Publication:**

  Wright, L.G., Onodera, T., Stein, M.M. et al. Deep physical neural networks trained with backpropagation. *Nature* **601**, 549–555 (2022). https://doi.org/10.1038/s41586-021-04223-6

All code is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. 

The full license text is provided in `LICENSE.txt`.
