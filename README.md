# LearningToAssemble
Learning to Assemble with Alternative Plans (SIGGRAPH 2025)

## Prerequisites
1. uv package manager
2. Gurobi License (optional)

## Installation
1. In the project folder, create a python virtual environment
```bash
uv venv --python=3.10
```
2. Activate the environment
3. Install the required packages
```bash
uv pip install -e .
```
4. Verify the test code
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
