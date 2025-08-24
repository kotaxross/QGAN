# QGAN

## Overview
This project implements a **Quantum Generative Adversarial Network (QGAN)** as described in the paper [Quantum generative adversarial network for generating discrete distribution](https://arxiv.org/abs/1807.01235), mainly using PyTorch and Qiskit.
A parameterized quantum circuit acts as the generator to produce discrete data, which is then distinguished from real data (Bars and Stripes, BAS patterns) by the discriminator in a GAN-based learning framework.

## Configuration
- `config.yaml`: Basic experimental settings.
- `experiment/<dir>/param.yaml`: Experimental conditions and parameters for comparative studies.

## Code Structure
- `dataset.py`: Provides BAS pattern data.
- `model.py`: Defines the parameterized quantum circuit (generator) and the discriminator.
- `utils.py`: Data conversion for GPU usage.  
- `main.py`: The main entry point for running experiments.

## Requirements
All required libraries are listed in `requirements.txt`.

## Running with Docker
You can easily reproduce the environment and run experiments with Docker.

### Build the image
```bash
docker build -t <image> .
```

### Run experiments
```bash
docker run --gpus all -v $PWD:/workspace <image> \
    python3 main.py -o experiment/<dir>
```
`main.py` reads the `config.yaml` located in the current directory and the `param.yaml` located in the specified output directory (`-o`).
Training is executed base on these configurations, and the log file as well as the training results are saved under the specified output directory.

## License
This project is licensed under the MIT License.
