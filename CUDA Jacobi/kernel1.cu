#include <stdio.h>
#include "kernel1.h"


extern  __shared__  float sdata[];

////////////////////////////////////////////////////////////////////////////////
//! Weighted Jacobi Iteration
//! @param g_dataA  input data in global memory
//! @param g_dataB  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void k1( float* g_dataA, float* g_dataB, int floatpitch, int width) 
{
    extern __shared__ float s_data[];
    // TODO, implement this kernel below

	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x + 1;
	unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y + 1;

	// setup block data
	if(idx < width && idy < width && idx > 0 && idy > 0) {
		if(1 == threadIdx.x && 1 == threadIdx.y) {
			s_data[0] = g_dataA[idy*width];
			s_data[blockDim.x] = g_dataA[idy*width + blockDim.x];
			s_data[2*blockDim.x] = g_dataA[idy*width + 2*blockDim.x];
		} else if ( width-1 == threadIdx.x && width-1 == threadIdx.y ) {
			s_data[blockDim.x-1] = g_dataA[idy*width+ blockDim.x -1];
			s_data[2*blockDim.x-1] = g_dataA[idy*width + 2*blockDim.x -1];
			s_data[3*blockDim.x-1] = g_dataA[idy*width + 3*blockDim.x -1];
		}

		sdata[threadIdx.x] = g_dataA[threadIdx.x];
		sdata[threadIdx.x + blockDim.x] = g_dataA[idy*width + blockDim.x + threadIdx.x];
		sdata[threadIdx.x + 2*blockDim.x] = g_dataA[idy*width + 2*blockDim.x +threadIdx.x];
	}
    
	__syncthreads();

	// perform jacobi pass
	if(idx < width && idy < width && idx > 0 && idy > 0) {
		sdata[threadIdx.x + 3*blockDim.x] = (
								0.2f * sdata[threadIdx.x + blockDim.x] +		// itself
								0.1f * sdata[threadIdx.x] +						// N
								0.1f * sdata[threadIdx.x+1] +					// NEz
								0.1f * sdata[threadIdx.x+blockDim.x+1] +		// E
								0.1f * sdata[threadIdx.x+(2*blockDim.x)+1] +	// SE
								0.1f * sdata[threadIdx.x+(2*blockDim.x)] +		// S
								0.1f * sdata[threadIdx.x+(2*blockDim.x)-1] +	// SW
								0.1f * sdata[threadIdx.x+blockDimx.x-1] +		// W
								0.1f * sdata[threadIdx.x-1] +					// NW
							) * 0.95f;
	
	
	}

	// Not sure this call is needed
	__syncthreads();

	if(idx < width && idy < width && idx > 0 && idy > 0) {
		g_dataB[idy*width+idx] = sdata[threadIdx.x + 3*blockDim.x];	
	}
}

