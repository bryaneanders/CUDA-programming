
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "pgmUtility.h"
#include "pgmProcess.h"

#define BLOCK_SIZE 32


// Implement or define each function prototypes listed in pgmUtility.h file.
// NOTE: You can NOT change the input, output, and argument type of the functions in pgmUtility.h
// NOTE: You can NOT change the prototype of any functions listed in pgmUtility.h


/**
 *  Function Name: 
 *      pgmRead()
 *      pgmRead() reads in a pgm image using file I/O, you have to follow the file format. All code in this function are exectured on CPU.
 *      
 *  @param[in,out]  header  holds the header of the pgm file in a 2D character array
 *                          After we process the pixels in the input image, we write the origianl 
 *                          header (or potentially modified) back to a new image file.
 *  @param[in,out]  numRows describes how many rows of pixels in the image.
 *  @param[in,out]  numCols describe how many pixels in one row in the image.
 *  @param[in]      in      FILE pointer, points to an opened image file that we like to read in.
 *  @return         If successful, return all pixels in the pgm image, which is an int **, equivalent to
 *                  a 2D array. Otherwise null.
 *
 */
int ** pgmRead( char **header, int *numRows, int *numCols, FILE *in  )
{
	char *toks, temp[maxSizeHeadRow+1];
	int i, j, **values;
	
	// not totally sure that this allocation will work properly with the 2d.
	// The header i pass in will already be allocated, but each of its elements won't be
	for( i = 0; i < rowsInHeader; i++){
		if( fgets(temp, maxSizeHeadRow, in) == NULL){
			printf("Improper file header\n");
			exit(EXIT_FAILURE);
		} 
		if(temp[strlen(temp)-1] == '\n') {
			temp[strlen(temp)-1] = '\0';
		}
	
		header[i] = (char*) calloc(strlen(temp)+1, sizeof(char));
		strncpy(header[i], temp, strlen(temp));
	}
	
	// get the number of rows and columns from header[2] using strok
	char str[strlen(header[2])+1];
	strncpy(str, header[2], strlen(header[2]));

	toks = strtok(str, " ");
	*numCols = atoi(toks);
	toks = strtok(NULL, " \n\0");
	*numRows = atoi(toks);

	values = (int**) malloc(*numRows * sizeof(int*));

	
	if(values == NULL){
		printf("Failed to allocate array of of color values\n");
		exit(EXIT_FAILURE);
	}
	
	for(i = 0; i < *numRows; i++) {
		values[i] = (int*) calloc(*numCols, sizeof(int));
		
		if(values[i] == NULL){
			printf("Failed to allocate array of of color values\n");
			exit(EXIT_FAILURE);
		}
	}


	// read the numbers out of the file and into the int array
	// need to make this 2d
	for(i = 0; i < *numRows; i++){ 
		for( j = 0; j < *numCols; j++){
					
			fscanf(in, "%d", &values[i][j]); 
						
		}
	}
	return values;
}


/**
 *  Function Name:
 *      pgmDrawCircle()
 *      pgmDrawCircle() draw a circle on the image by setting relavant pixels to Zero.
 *                      In this function, you have to invoke a CUDA kernel to perform all image processing on GPU.
 *
 *  @param[in,out]  pixels  holds all pixels in the pgm image, which a 2D integer array. The array
 *                          are modified after the drawing.
 *  @param[in]      numRows describes how many rows of pixels in the image.
 *  @param[in]      numCols describes how many columns of pixels in one row in the image.
 *  @param[in]      centerCol specifies at which column you like to center your circle.
 *  @param[in]      centerRow specifies at which row you like to center your circle.
 *                        centerCol and centerRow defines the center of the circle.
 *  @param[in]      radius    specifies what the radius of the circle would be, in number of pixels.
 *  @param[in,out]  header returns the new header after draw. 
 *                  the circle draw might change the maximum intensity value in the image, so we
 *                  have to change maximum intensity value in the header accordingly.
 *  @return         return 1 if max intensity is changed, otherwise return 0;
 
*/
int pgmDrawCircle( int **pixels, int numRows, int numCols, int centerRow,
                   int centerCol, int radius, char **header )
{
	int i, *d_pixels, max_changed = 0;

	cudaMalloc((void**) &d_pixels, sizeof(int) * numRows * numCols);

	// copy the data in the pixel array from the host to the device
	for( i = 0; i < numRows; i++ ){
		cudaMemcpy(d_pixels+(i*numCols), pixels[i], sizeof(int) * numCols, cudaMemcpyHostToDevice);
	}

	// invoke the kernel
	// invokeCircleKernel(&d_pixels, numRows, numCols, centerRow, centerCol, radius);
	// set up the kernel's config structs
	dim3 grid, block;
    block.x = BLOCK_SIZE;
    block.y = BLOCK_SIZE;
    grid.x  = ceil((float)numCols / block.x);
    grid.y  = ceil((float)numRows / block.y);

	// launch the kernel
	draw_circle<<<grid, block>>>(d_pixels, numRows, numCols, centerRow, centerCol, radius);
	
	for( i = 0; i < numRows; i++ ){
		cudaMemcpy(pixels[i], (d_pixels)+(i*numCols), sizeof(int) * numCols, cudaMemcpyDeviceToHost);
	}

	cudaFree(d_pixels);
	
	// find the max value in the now modified pixels array and check it against the value in the header
	// if it changed update max_changed
	int new_max = find_max_intensity(pixels, numRows, numCols);
	
	int curr_max = atoi(header[3]);
	
	if(new_max > curr_max) {
		char temp[10];
		snprintf(temp, 10, "%d", new_max); 
		
		strncpy(header[3], temp, strlen(temp));
	}
	
	// return whether or not the maximum changed
	return max_changed;
}

/**
 *  Function Name:
 *      pgmDrawEdge()
 *      pgmDrawEdge() draws a black edge frame around the image by setting relavant pixels to Zero.
 *                    In this function, you have to invoke a CUDA kernel to perform all image processing on GPU.
 *
 *  @param[in,out]  pixels  holds all pixels in the pgm image, which a 2D integer array. The array
 *                          are modified after the drawing.
 *  @param[in]      numRows describes how many rows of pixels in the image.
 *  @param[in]      numCols describes how many columns of pixels in one row in the image.
 *  @param[in]      edgeWidth specifies how wide the edge frame would be, in number of pixels.
 *  @param[in,out]  header returns the new header after draw.
 *                  the function might change the maximum intensity value in the image, so we
 *                  have to change the maximum intensity value in the header accordingly.
 *
 *  @return         return 1 if max intensity is changed by the drawing, otherwise return 0;
 */
int pgmDrawEdge( int **pixels, int numRows, int numCols, int edgeWidth, char **header )
{
	int i, *d_pixels, max_changed = 0;
	
	// copy the data in the pixel array from the host to the device
	cudaMalloc((void**) &d_pixels, sizeof(int) * numRows * numCols);

	// copy the data in the pixel array from the host to the device
	for( i = 0; i < numRows; i++ ){
		cudaMemcpy(d_pixels+(i*numCols), pixels[i], sizeof(int) * numCols, cudaMemcpyHostToDevice);
	}

	// invoke the kernel
	// invokeEdgeKernel(&d_pixels, numRows, numCols, edgeWidth);

	// set up the kernel's config structs
	dim3 grid, block;
    block.x = BLOCK_SIZE;
    block.y = BLOCK_SIZE;
    grid.x  = ceil((float)numCols / block.x);
    grid.y  = ceil((float)numRows / block.y);

	// launch the kernal
	draw_edge<<<grid, block>>>(d_pixels, numRows, numCols, edgeWidth);
	
	for( i = 0; i < numRows; i++ ){
		cudaMemcpy(pixels[i], d_pixels+(i*numCols), sizeof(int) * numCols, cudaMemcpyDeviceToHost);
	}

	cudaFree(d_pixels);

	// find the max value in the now modified pixels array and check it against the value in the header
	// if it changed update max_change
	int new_max = find_max_intensity(pixels, numRows, numCols);
	
	int curr_max = atoi(header[3]);
	
	if(new_max > curr_max) {
		char temp[10];
		snprintf(temp, 10, "%d", new_max); 
		
		strncpy(header[3], temp, strlen(temp));
	}

	cudaFree(d_pixels);
	
	// return whether or not the maximum changed
	return max_changed;

}
/*
void printArray(int** a, int rows, int cols)
{
	int i, j;
	for( i = 0; i < rows; i++ ){
		for( j = 0; j < cols; j++ ){
		
		}
	}
	
}*/

/**
 *  Function Name:
 *      pgmWrite()
 *      pgmWrite() writes headers and pixels into a pgm image using file I/O.
 *                 writing back image has to strictly follow the image format. All code in this function are exectured on CPU.
 *
 *  @param[in]  header  holds the header of the pgm file in a 2D character array
 *                          we write the header back to a new image file on disk.
 *  @param[in]  pixels  holds all pixels in the pgm image, which a 2D integer array.
 *  @param[in]  numRows describes how many rows of pixels in the image.
 *  @param[in]  numCols describe how many columns of pixels in one row in the image.
 *  @param[in]  out     FILE pointer, points to an opened text file that we like to write into.
 *  @return     return 0 if the function successfully writes the header and pixels into file.
 *                          else return -1;
 */
int pgmWrite( const char **header, const int **pixels, int numRows, int numCols, FILE *out )
{
	int i, j, k, difference, len = max_intensity_digits(atoi(header[3]))+1;
	if( out == NULL ){
		printf("No file pointer passed to pgmWrite\n");
		return -1;
	}
	
	for(i = 0; i < rowsInHeader; i++ ){
		fprintf(out, "%s\n", header[i]);
	}
	
	for( i = 0; i < numRows; i++ ){
		for(j = 0; j < numCols; j++){
			difference = len - max_intensity_digits(pixels[i][j]);

			fprintf(out, "%d", pixels[i][j]);

			for( k = 0; k < difference; k++) {
				fprintf(out, " ");
			}
		}
		fprintf(out, "\n");


	}

	return 0;
}

/**
 *  Function Name:
 *		find_max_intensity()
 *		find_max_intensity() looks through a 2d array of integers, a, and returns the largest int value in the array.
 *
 *	@param[in]	a		a 2d integer array which holds the intensity values we will check
 *  @param[in]	rows	an integer which denotes the number of rows in a
 *  @param[in]	cols	an integer which denotes the number of columns in a
 *  @return		returns a default of 0, black, which is replaced by the largest, brightest, integer value in the array.
 */
int find_max_intensity(int **a, int rows, int cols)
{
	int i, j, max = 0;
	
	for(i = 0; i < rows; i++) {
		for(j = 0; j < cols; j++) {

			if(a[i][j] > max) {
				max = a[i][j];
			}
		}
	}
	
	return max;
}


/**
 *  Function Name:
 *		max_intensity_digits()
 *		max_intensity_digits() Determines the number of digits in a positive integer
 *
 *	@param[in]	n	The int whose length in digits will be determined	
 *
 *  @return			returns the number of digis in n if n is less than 1 million. If n is negative it returns 0.
 */
int max_intensity_digits(int n)
{
	if(n < 0) {
		printf("intensity must be positive");
		return 0;
	}
	if(n >= 10){
		if(n >=	100) {
			if( n >= 1000 ){
				if( n >= 10000 ){
					if( n >= 100000 ) {
						return 6;
					}
					return 6;
				}	
				return 4;
			}
			return 3;
		}

		return 2;		
	}

	return 1;
}

