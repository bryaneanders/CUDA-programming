// my kernel
__global__ void naiveMergeSortTileKernel(
	unsigned int *d_DstKey,
	unsigned int *d_DstVal,
	unsigned int *d_SrcKey,
	unsigned int *d_SrcVal,
	unsigned int arrayLength,
	unsigned int stride, 
	unsigned int sortDir
);


// Bottom-level merge sort (binary search-based), this is 
__global__ void mergeSortSharedKernel(
    unsigned int *d_DstKey,
    unsigned int *d_DstVal,
    unsigned int *d_SrcKey,
    unsigned int *d_SrcVal,
    unsigned int arrayLength,
	unsigned int totalArrayLen,
	unsigned int sortDir
);

__device__ unsigned int binarySearchExclusive(unsigned int val, unsigned int *data, unsigned int L, unsigned int stride, unsigned int sortDir);


__device__ unsigned int binarySearchInclusive(unsigned int val, unsigned int *data, unsigned int L, unsigned int stride, unsigned int sortDir);
