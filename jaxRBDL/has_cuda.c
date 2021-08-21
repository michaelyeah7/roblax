#include <cuda_runtime.h>
#include "helper_cuda.h"
int main ()
{
    int deviceCount;
    cudaError_t e = cudaGetDeviceCount(&deviceCount);
    return e == cudaSuccess ? deviceCount : -1;
}
