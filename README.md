# Code for Robust-Region-Feature-Synthesizer-for-Zero-Shot-Object-Detection
### 1. Environment requirements
#### - [mmdetection](http://github.com/open-mmlab/mmdetection) we recommend using [Docker 2.0](Docker.md). 
#### - The code implementation of our experiments mainly based on [PyTorch 1.1.0](https://pytorch.org/) and Python 3.6.
#### - The following scripts are for dfferents steps in the pipeline on PASCAL VOC dataset, please see the respective files for more arguments. Before running the scripts, please set the datasets and backbone paths in the config files. Weights of [ResNet101](https://drive.google.com/file/d/1g3UXPw-_K3na7acQGZlhjgQPjXz_FNnX/view?usp=sharing) trained excluding overlapping unseen classes from ImageNet.
