# cuda_codes
GEMM:
  nvcc -m64 -arch sm_75 -o test gemm.cu -lcublas
  ./test 512 256 512