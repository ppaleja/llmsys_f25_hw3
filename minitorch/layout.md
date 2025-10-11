minitorch/                  # The minitorch source code
    autodiff.py             # Automatic differentiation implementation
    cuda_kernel_ops.py      # Connects Tensor backend with the CUDA kernels
    tensor.py               # Defines the Tensor class
    tensor_data.py          # Defines the TensorData class
                              used in the Tensor class
    tensor_functions.py     # Defines the functions
                              used in the Tensor class
    tensor_ops.py           # Defines the TensorBackend class
                              used in the Tensor class
src/
    combine.cu              # CUDA kernels implementation
project/
    run_sentiment.py        # Network and training codes for training for
                              the sentence sentiment classification task
