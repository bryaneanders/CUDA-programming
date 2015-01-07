
#include<stdio.h>
#include "mergeSortKernels.h"


// my kernel, sorts the already sorted tiles into
// an entirely merge sorted whole
__global__ void naiveMergeSortTileKernel(
	unsigned int *d_DstKey,
	unsigned int *d_DstVal,
	unsigned int *d_SrcKey,
	unsigned int *d_SrcVal,
	unsigned int arrayLength,
	unsigned int stride,
	unsigned int sortDir
) 
{
	unsigned int keyA, keyB, valA, valB, posA, posB;
	unsigned int cIndex;
		
	// arrray index for processing at the start of the block
	cIndex = ((gridDim.x * blockIdx.y) + blockIdx.x)*(stride*2);
		
	d_DstKey += cIndex;
	d_DstVal += cIndex;
	d_SrcKey += cIndex;
	d_SrcVal += cIndex;
		
	// If a tile has less than stride elements, it does NOTHING
		
	for(unsigned int i = 0; i < stride / blockDim.x; i++) {
	
		__syncthreads();
				
		if ( cIndex + stride*2 <= arrayLength) { 		
		
			if( cIndex + blockDim.x*i + threadIdx.x < arrayLength ) {
				keyA = d_SrcKey[threadIdx.x + blockDim.x*i];
				valA = d_SrcVal[threadIdx.x + blockDim.x*i];
				posA = binarySearchExclusive(keyA, d_SrcKey +stride ,  stride, stride,  sortDir)
										 +blockDim.x*i + threadIdx.x;
			}
			
			if( cIndex + stride + blockDim.x*i + threadIdx.x < arrayLength ) {
				keyB = d_SrcKey[stride + threadIdx.x + blockDim.x*i];
				valB = d_SrcVal[stride + threadIdx.x + blockDim.x*i];
				posB = binarySearchInclusive(keyB, d_SrcKey + 0, stride, stride, sortDir)
										+ blockDim.x*i + threadIdx.x;
			} 
		}
		__syncthreads();
			
		if( cIndex + 2*stride <= arrayLength) {
			if( cIndex + blockDim.x*i + threadIdx.x < arrayLength ) {
				d_DstKey[posA] = keyA;
				d_DstVal[posA] = valA;
			}
			if( cIndex + stride + blockDim.x*i + threadIdx.x < arrayLength ) {
				d_DstKey[posB] = keyB;
				d_DstVal[posB] = valB;
			}
		}
	}
}


			
// ** NOTE: this includes code that makes it so that this kernel
// does not require a full block, but my final implementation 
// will ensure that it does

// Bottom-level merge sort (binary search-based), this is 
__global__ void mergeSortSharedKernel(
    unsigned int *d_DstKey,
    unsigned int *d_DstVal,
    unsigned int *d_SrcKey,
    unsigned int *d_SrcVal,
    unsigned int arrayLength,   //actual length for this block to process in the input array.
	unsigned int totalArrayLen,
	unsigned int sortDir 	
)
{
	// the shared memory is 4x the size of blockDim.x/
	// the second half of the shared memory is assigned to s_val
    extern __shared__ unsigned int s_key[];   
	
	unsigned int memSize = blockDim.x*2;	
	
	// the start location for a thread in a 2d grid where each
	// thread processes 2 elements
	unsigned int gridBlock = ((gridDim.x * blockIdx.y) + blockIdx.x)*memSize;
	unsigned int tid = gridBlock + threadIdx.x;

    d_SrcKey += tid;
    d_SrcVal += tid;
    d_DstKey += tid;
    d_DstVal += tid;

	unsigned int *s_val;

	if( gridBlock + memSize<= totalArrayLen ){
		s_val = s_key + memSize;
		s_key[threadIdx.x + 0] = d_SrcKey[0];
		s_val[threadIdx.x + 0] =  d_SrcVal[0];
		s_key[threadIdx.x + memSize/2 ] = d_SrcKey[memSize/2];
		s_val[threadIdx.x + memSize/2 ]  = d_SrcVal[memSize/2];
	}

	unsigned int lPos, *baseKey, *baseVal;
	unsigned int keyA, keyB, valA, valB, posA, posB;
	for (unsigned int stride = 1; stride < arrayLength; stride <<= 1)
	{		
		if( gridBlock + memSize<= totalArrayLen ){
			lPos = threadIdx.x  &  (stride - 1);
			baseKey = s_key + 2 * (threadIdx.x - lPos);
			baseVal = s_val + 2 *   (threadIdx.x - lPos);
		}
		__syncthreads();
			
		// get the thread's 2 elements and find their
		// position in the output global arrays
		if( gridBlock + memSize<= totalArrayLen ){
			keyA = baseKey[lPos +      0];
			valA = baseVal[lPos +      0];
			keyB = baseKey[lPos + stride];
			valB = baseVal[lPos + stride];
			posA = binarySearchExclusive(keyA, baseKey + stride, stride, stride, sortDir) + lPos;
			posB = binarySearchInclusive(keyB, baseKey + 	   0, stride, stride, sortDir) + lPos;  
		}
		__syncthreads();
			
		// inset the key and val elements into shared memory
		if( gridBlock + memSize<= totalArrayLen ){	
			baseKey[posA] = keyA;
			baseVal[posA] = valA;
			baseKey[posB] = keyB;
			baseVal[posB] = valB;
		}
	}

	__syncthreads();
	// copy data from shared to global memory if the block
	// is not the tail incomplete block
	if( gridBlock + memSize<= totalArrayLen ){
		d_DstKey[              0] = s_key[threadIdx.x +     0];
		d_DstVal[               0]  = s_val[threadIdx.x +     0];
		d_DstKey[memSize/2] = s_key[threadIdx.x + memSize/2 ];
		d_DstVal[memSize/2] =   s_val[threadIdx.x + memSize/2 ];
	} else if(gridBlock + memSize > totalArrayLen && tid < totalArrayLen) {
		// Sort out the last partial block if it exists
		// this is not efficient, but i'm strapped for time

	   // set the addresses back to the start of the block
		d_SrcKey -= threadIdx.x;
		d_SrcVal -= threadIdx.x;
		d_DstKey -= threadIdx.x;
		d_DstVal -= threadIdx.x;
		unsigned int remArrayLen = totalArrayLen-gridBlock;
		unsigned int count = 0;
		
		// count how many elements go before this thread's in this
		// section of global memory
		for(unsigned int i = 0; i < remArrayLen; i++) {
			if(d_SrcKey[threadIdx.x] > d_SrcKey[i]) {
				count++;
			} else if(d_SrcKey[threadIdx.x] == d_SrcKey[i] && threadIdx.x > i){
				count++;
			}
		}
			
		d_DstKey[count] = d_SrcKey[threadIdx.x];
		d_DstVal[count] = d_SrcVal[threadIdx.x];
		}
}

// from the CUDA package
__device__ unsigned int binarySearchInclusive
(
unsigned int val,
unsigned int *data, 
unsigned int L, 
unsigned int stride, 
unsigned int sortDir
)
{
    if (L == 0) {
        return 0;
    }

    unsigned int pos = 0;
    for (; stride > 0; stride >>= 1){
        unsigned int newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] <= val)) || (!sortDir && (data[newPos - 1] >= val))) {
            pos = newPos;
        }
    }
    return pos;
}

// also from the CUDA package
__device__ unsigned int binarySearchExclusive
(
unsigned int val, 
unsigned int *data, 
unsigned int L, 
unsigned int stride, 
unsigned int sortDir
)
{
    if (L == 0) {
        return 0;
    }

    unsigned int pos = 0;
    for (; stride > 0; stride >>= 1){
        unsigned int newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val))){
            pos = newPos;
        }
    }
    return pos;
}



