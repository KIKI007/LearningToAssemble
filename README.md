# LearningToAssemble
Learning to Assemble with Alternative Plans (SIGGRAPH 2025)

## Prerequisites
1. Pytorch 2.4.0 + Cuda 12.4 (currently because of PyG)
2. Gurobi License (optional)

## Installation
In the project folder
```bash
pip install -e .
```
Then, to verify the code
```bash
pytest .
```
Error may occur if you have not obtained Gurobi License. However, this does not affect the training.

## Training & Testing
```bash
python script/train tetris-1
```
```bash
python script/test tetris-1
```
For more options please review the code.
