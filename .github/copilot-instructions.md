# Copilot Instructions for llmsys_f25_hw3

## Project Overview
This project implements a mini deep learning framework and transformer models for a course assignment. The codebase is organized into modular components for tensor operations, neural network layers, CUDA acceleration, and end-to-end NLP pipelines.

## Key Components
- **minitorch/**: Core framework code. Includes tensor operations (`tensor_ops.py`), scalar/tensor functions, modules (`module.py`, `modules_basic.py`, `modules_transfomer.py`), CUDA support (`cuda_kernel_ops.py`, `cuda_ops.py`), and testing utilities (`testing.py`).
- **project/**: Scripts for running sentiment analysis and machine translation pipelines.
- **src/combine.cu**: CUDA kernel implementations for accelerated tensor ops.
- **tests/**: Pytest-based unit tests for all major modules. Data-driven tests use `.npy` files in `tests/data/`.


## Developer Workflows

**Important:** All program runs (tests, scripts, etc.) are executed on a separate server. If you need to run tests or scripts, provide the exact command to the user and wait for them to return the output. Do not attempt to execute code directly.

- **Install & Setup**:
  - `pip install -r requirements.extra.txt && pip install -r requirements.txt`
  - `pip install -e .` (editable install for development)
  - Compile CUDA kernels: `bash compile_cuda.sh`
- **Testing**:
  - To run all tests: `python -m pytest -l -v`
  - To run specific tests (e.g., for modules):
    - `python -m pytest -l -v -k "test_linear_student"`
    - `python -m pytest -l -v -k "test_dropout_student"`
    - `python -m pytest -l -v -k "test_layernorm_student"`
    - `python -m pytest -l -v -k "test_embedding_student"`
- **Run End-to-End Pipelines**:
  - Sentiment: `python project/run_sentiment_linear.py`
  - Machine Translation: `python project/run_machine_translation.py`

## Project-Specific Conventions
- **Assignment Markers**: Code sections to implement are marked with `# ASSIGN...` and `# END ASSIGN...` comments.
- **Backend Abstraction**: Many modules accept a `backend` argument to switch between CPU and CUDA implementations.
- **Testing**: Uses both property-based (Hypothesis) and data-driven (NumPy `.npy` files) tests. See `tests/` and `minitorch/testing.py` for patterns.
- **CUDA Integration**: CUDA kernels are defined in `src/combine.cu` and mapped in `minitorch/cuda_kernel_ops.py`. Always recompile after kernel changes.
- **Module Structure**: All neural network layers inherit from `minitorch.module.Module`.
- **Randomness**: For reproducibility, use `np.random.binomial` for dropout masks as specified in the README.

## Integration Points
- **Tensor Operations**: Implemented in `minitorch/tensor_ops.py` and `minitorch/cuda_kernel_ops.py`.
- **Operators**: Scalar ops in `minitorch/operators.py` are used throughout the codebase and mapped to CUDA kernels.
- **Datasets**: Synthetic datasets for testing are in `minitorch/datasets.py`.

## Examples
- To add a new tensor operation, update both `tensor_ops.py` (CPU) and `cuda_kernel_ops.py` (CUDA), and map the function in `src/combine.cu`.
- To implement a new module, inherit from `Module` and follow patterns in `modules_basic.py` or `modules_transfomer.py`.

## References
- See `README.md` for assignment details, setup, and workflow instructions.
- Test data and expected outputs are in `tests/data/`.

---
For any unclear or missing conventions, please request clarification or check the README for assignment-specific instructions.
