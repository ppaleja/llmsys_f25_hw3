mkdir -p minitorch/cuda_kernels
nvcc -O2 -arch=sm_75 -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
