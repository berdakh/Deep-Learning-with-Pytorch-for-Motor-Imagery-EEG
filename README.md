# Deep Learning with PyTorch for Motor Imagery EEG

This repository provides a comprehensive framework for analyzing motor imagery EEG data using deep learning techniques implemented in PyTorch. It includes data preprocessing, model training, evaluation, and visualization tools to facilitate research and development in brain-computer interfaces (BCIs).

## Features

* **Data Preprocessing**: Scripts for loading and preprocessing EEG data using MNE-Python.
* **Model Training**: Jupyter notebooks for training convolutional neural networks (CNNs) on pooled and subject-specific datasets.
* **Evaluation**: Tools for evaluating model performance across different training paradigms.
* **Visualization**: Notebooks for visualizing EEG data and model outputs to aid in analysis.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/berdakh/Deep-Learning-with-Pytorch-for-Motor-Imagery-EEG.git
   cd Deep-Learning-with-Pytorch-for-Motor-Imagery-EEG
   ```

2. **Create a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:

   ```bash
   pip install mne==0.18.0
   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
   pip install braindecode
   ```

   *Note: Ensure that [MNE-Python](https://mne.tools/stable/index.html) and [PyTorch](https://pytorch.org/) are installed, as they are central to the functionalities provided.*

## Usage

The repository includes several scripts and notebooks to facilitate different stages of the EEG analysis pipeline:

* **Data Preprocessing**:

  * `nu_smrutils.py`: Functions for loading and preprocessing EEG data.

* **Model Training**:

  * `train_CNN_pooled.ipynb`: Notebook for training CNN models on pooled datasets.
  * `train_CNN_subspe.ipynb`: Notebook for training CNN models on subject-specific datasets.
  * `train_method.ipynb`: Notebook demonstrating various training methods and comparisons.

* **Visualization**:

  * `visualization.ipynb`: Notebook for visualizing EEG data and model outputs.

*To execute a notebook, navigate to the repository directory and run:*

```bash
jupyter notebook
```

*Then open the desired notebook (e.g., `train_CNN_pooled.ipynb`) to begin.*

## Dataset

The implementation utilizes EEG datasets suitable for motor imagery analysis. Ensure that your dataset is properly formatted and placed in the appropriate directory. Update the paths in the scripts as necessary.
 
