# LearningToAssemble
Learning to Assemble with Alternative Plans (SIGGRAPH 2025)

## Option 1. Installation using uv
1. In the project folder, create a python virtual environment
```bash
uv venv --python=3.10
```
2. Activate the environment
3. Install Pytorch
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 # 40 series
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 # 50 series
```
4. Install the package
```bash
uv pip install -e .
```
5. Verify the test code
```bash
pytest .
```
Error may occur if you have not obtained Gurobi License. However, this does not affect the training.

### Option 2. Installation inside a docker container
1. Install Pytorch (this step can be skipped if container has the right version of pytorch)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 # 40 series
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 # 50 series
```
2. Install the package
```bash
pip install -e .
```
3. Verify the test code
```bash
pytest .
```

## Training & Testing
```bash
python script/train tetris-1
```
```bash
python script/test tetris-1
```
For more options please review the code.
