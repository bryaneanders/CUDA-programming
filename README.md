CUDA-programming
================

Some programs that run on NVIDIA gpus. Portions of the code, particularly some method stubs, written by Yun Tian at Eastern Washington University.

Programs:

1) C serial and parallel word sort and count. Uses merge sort, both serial and parallel.
2) A Jacobi Itteration performed on the GPU. Some method stubs by Yun Tian.
3) Merge sort on the GPU. Tiles are sorted by an algorithm provided in the CUDA code, 
    my code sorts the sorted tiles into a more cohesively sorted whole. Also capable of 
    running a serial merge sort for comparison.
4) Simple image processing on the gpu. Draws a border or circle on a size specified by by
    command link arguments on the image chosen. Some method stubs by Yun Tian.
    
    