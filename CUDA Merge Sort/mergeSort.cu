
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mergeSort.h"
#include "mergeSortKernels.h"
#include "mergeSortSerial.h"
#include "timing.h"

#define ROLLMAX 10001

int main(int argc, char *argv[])
{
	cudaDeviceReset();

	int shouldPrint = 0;
	
	// to run this program use:
	// ./mergeSort numInts blockWidth p
	if(argc < 3 || argc > 4)
	{
		usage();
		return 1;
	} else if (argc == 3) {
		shouldPrint = 0;
	} else if (argv[3][0]=='p') {
		shouldPrint = 1;
	} else {
		usage();
		return 1;
	}

	unsigned int numInts = atoi(argv[1]);
	unsigned int blockWidth = atoi(argv[2]);
	unsigned int runCpu;
	if(!blockWidth) {
		runCpu = 1;
	}
	
	if(blockWidth > 0 && numInts  % (blockWidth*2) != 0) {
		printf("numInts must be a multiple of blockWidth*2\n");
		return -1;
	}
	
	size_t bytes = numInts * sizeof(unsigned int);
	
	srand(time(NULL));

	unsigned int *h_key, *h_val, *h_dstVal, *h_dstKey;
	unsigned int *d_dstKey, *d_dstVal, *d_srcKey, *d_srcVal;
	
	// allocate host memory
	h_key = (unsigned int*) calloc(numInts, sizeof(unsigned int));
	h_val = (unsigned int*) calloc(numInts, sizeof(unsigned int));
	h_dstKey = (unsigned int*) calloc(numInts, sizeof(unsigned int));
	h_dstVal = (unsigned int*) calloc(numInts, sizeof(unsigned int));
	
	if(!h_key || !h_val || !h_dstKey || !h_dstVal  ) 
	{
		printf("Host Memory allocation failed\n");
		exit(-1);
	}
	
		// allocate host source key and value arrays
	fillKeyArrayRandom(h_key, numInts);
	fillValArray(h_val, numInts);
	
	if(!runCpu){
		// allocate device memory
		cudaMalloc((void**) &d_srcKey, bytes);
		cudaMalloc((void**) &d_srcVal, bytes);
		cudaMalloc((void**) &d_dstKey, bytes);
		cudaMalloc((void**) &d_dstVal, bytes); 
		
		if(!d_dstKey || !d_dstVal || !d_srcKey || !d_srcVal ) 
		{
			printf("Device Memory allocation failed\n");
			exit(-1);
		}
	
	}
	
	// print the sorted results if requested
	if(shouldPrint) {
		printArrays(h_key, h_val, numInts);
	}		
	
	if(!runCpu){
		// copy memory from the host to the device source arrays
		cudaMemcpy(d_srcKey, h_key, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_srcVal, h_val, bytes, cudaMemcpyHostToDevice);
	}

	// call the MSITR kernel
	if (numInts < 2)
    {
        return -1;
    }
	
	if(!runCpu){
		//unsigned int batchSize = 1;
		unsigned int tileWidth = 2*blockWidth;	
		
		// 2 elements per thread in the block, 
		unsigned int memsize = tileWidth * sizeof(unsigned int) * 2;
		unsigned int numTotalBlocks = ceil((float)numInts / tileWidth);
		unsigned int numBlocksy = ceil((float)numTotalBlocks / 65535);
		unsigned int numBlocksx  = ceil((float)numTotalBlocks / numBlocksy);

		// set up the dim3's to define grid and block size
		const dim3 blockSize(blockWidth, 1, 1);
		const dim3 numBlocks(numBlocksx, numBlocksy, 1);
		
		  // time the kernel launches using CUDA events
		cudaEvent_t launch_begin, launch_end;
		cudaEventCreate(&launch_begin);
		cudaEventCreate(&launch_end);
		cudaEventRecord(launch_begin,0);

		mergeSortSharedKernel<<<numBlocks, blockSize, memsize>>>(d_dstKey, d_dstVal, d_srcKey, d_srcVal, tileWidth, numInts, 1);

		// synchronize the device after the kernel call
		cudaDeviceSynchronize();
		
		cudaMemcpy(h_dstKey, d_dstKey, bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_dstVal, d_dstVal, bytes, cudaMemcpyDeviceToHost);
		
		// print the results of the BSITR kernel
		if(shouldPrint) {
			printArrays(h_dstKey, h_dstVal, numInts);
		}
		
		// swap the addresses of the input and output pointers
		// for the 2nd kernel call
		
		unsigned int *tempKey, *tempVal;
		
		for( unsigned int stride = tileWidth; stride < numInts; stride *= 2 ) {
			tempKey = d_dstKey;
			tempVal = d_dstVal;
			d_dstKey = d_srcKey;
			d_dstVal = d_srcVal;
			d_srcKey = tempKey;
			d_srcVal = tempVal;
			
			// naive tile merge kernel
			naiveMergeSortTileKernel<<<numBlocks, blockSize>>>(d_dstKey, d_dstVal, d_srcKey, d_srcVal, numInts, stride, 1);
			// synchronize the device after the kernel call
			cudaDeviceSynchronize();		
		}
		// record end time and time elapsed
		cudaEventRecord(launch_end,0);
		cudaEventSynchronize(launch_end);

		// measure the time spent in the kernel
		float time = 0;
		cudaEventElapsedTime(&time, launch_begin, launch_end);
		printf("GPU merge sort ran in %f milliseconds\n", time);


		// copy the results from the device to the host
		cudaMemcpy(h_dstKey, d_dstKey, bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_dstVal, d_dstVal, bytes, cudaMemcpyDeviceToHost);
	
		// print the sorted results if requested
		if(shouldPrint) {
			printArrays(h_dstKey, h_dstVal, numInts);
		}
		// free device memory
		cudaFree(d_srcKey);
		cudaFree(d_srcVal);
		cudaFree(d_dstKey);
		cudaFree(d_dstVal);
	}
	
	// serial merge sort
	if(runCpu){
		double then = currentTime();
		partition(&h_key, &h_val, 0,  numInts-1);
		double now = currentTime();
		float time = 0;
		time = (now - then) * 1000;
		printf("CPU code executed in %f milliseconds\n", time);
		
		if(shouldPrint) {
			printArrays(h_key, h_val, numInts);
		}
	}
	
	// free host memory
	free(h_val);
	free(h_key);
	free(h_dstKey);
	free(h_dstVal);
	
	return 0;
}

// fill an array with randomly generated unsigned int values
void fillKeyArrayRandom(unsigned int *keys, unsigned int numInts)
{
	for(int i = 0; i < numInts; i++){
		keys[i] = rand() % ROLLMAX;
	}
}

// fill the Value array with the index of the key array to which it coresponds
void fillValArray(unsigned int *vals, unsigned int numInts)
{
	for(int i = 0; i < numInts; i++) {
		vals[i] = i;
	}
}

// print the given arrays of keys and values of size size to the console
void printArrays(unsigned int *keys, unsigned int *vals, unsigned int size)
{
	unsigned int i;
	
	printf("keys: ");
	for(i = 0; i < size; i++) {
		printf("%u ", keys[i]);
		
		if( i % 32 == 0 && i > 0){
			printf("\n");
		}
	}
	
	printf("\nvals: ");
	for(i = 0; i < size; i++) {
		printf("%u ", vals[i]);
		
		if( i % 32 == 0 && i > 0){
			printf("\n");
		}
	}
	printf("\n\n");
}

// print the required args to the command line
void usage()
{
	printf("Usage: ./progName numInts blockWidth p\n");
}


