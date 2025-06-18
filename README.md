# SSD-LFM
SSD-LFM: Self-supervised denoising strategies for multi-modality light-field microscopy
This repository contains the official implementation of SSD-LFM, a self-supervised denoising framework for light field microscopy reconstruction. The code is organized into two main modules: data processing (MATLAB-based) and the reconstruction model (PyTorch-based).

REPOSITORY STRUCTURE
data_process/ - MATLAB scripts for dataset generation
tubulins_simulation.m - Simulates tubulin microscopy volumes
noise_adding.m - Adds realistic noise to simulated data
recorrupt_and_reconstruct.m - Reconstructs volumes using recorruption-based methods
data_process/temp/ - Output directory for processed datasets (auto-generated)

model/ - SSD-LFM network 
Main_train_s2s_3D.py - code for training
Main_test_s2s_3D - code for testing
command.py - demo
requirements.yml - Python dependencies


DATASET GENERATION
Execute these MATLAB scripts in order:

Volume Simulation (data_process/tubulins_simulation):
Output: temp/HR_raw_1

Noise Injection (data_process/noise_adding):
Output: temp/HR_noise_1

Recorruption-Based Reconstruction (data_process/recorrupt_and_reconstruct):
Output: temp/synthetic_tubulins

MODEL TRAINING/TESTING
Use the demo command to train/test the model

Please first change the path in line 88(training), line 68(testing)to your own

SUPPORT
Contact: zengxuyu@buaa.edu.cn 
