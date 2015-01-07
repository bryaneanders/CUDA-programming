# Build tools
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# here are all the objects
GPUOBJS = mergeSort.o mergeSortKernels.o 
OBJS = mergeSortSerial.o timing.o

# make and compile
mergeSort :$(OBJS) $(GPUOBJS)
	$(NVCC)  -g -G -o mergeSort $(OBJS) $(GPUOBJS) 

mergeSort.o: mergeSort.cu
	$(NVCC) -g -G -c mergeSort.cu 

mergeSortKernels.o: mergeSortKernels.cu
	$(NVCC) -g -G -c mergeSortKernels.cu

mergeSortSerial.o: mergeSortSerial.c
	$(CXX) -g -c mergeSortSerial.c
	
timing.o: timing.c
	$(CXX) -g -c timing.c

clean:
	rm -f *.o
	rm -f mergeSort
