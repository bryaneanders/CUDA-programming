#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pgmProcess.h"
#include "pgmUtility.h"

#define BLOCK_SIZE 32


/**
 *  Function Name:
 *      distance()
 *      distance() returns the Euclidean distance between two pixels. This function is executed on CUDA device
 *
 *  @param[in]  p1  coordinates of pixel one, p1[0] is for row number, p1[1] is for column number
 *  @param[in]  p2  coordinates of pixel two, p2[0] is for row number, p2[1] is for column number
 *  @return         return distance between p1 and p2
 */
__device__ float distance( int p1[], int p2[] )
{
	return sqrt(pow((float) p2[0]-p1[0], 2) + pow((float) p2[1] - p1[1], 2));
}


/**
 *  Function Name:
 *      draw_circle()
 *      draw_circle() draws a black circle centered on centerRow and Center col with the given radius.
 *					  Each call checks one element of the pixels array
 *
 *  @param[in]  pixels		a 2d int array that contains the intensity values of the image
 *  @param[in]  rows		the number of elements in a row of the pixels array
 *	@param[in]  cols 		the number of elements in a column of the pixels array
 *	@param[in]  centerRow	the row that the drawn circle will be cented on
 *	@param[in]	centerCol 	the column that the drawn circle will be centered on
 *	@param[in] 	radius		the size of the circle to be drawn around its center element
 *  @return     void
 */
__global__ void draw_circle(int *pixels, int rows, int cols, int centerRow, 
							int centerCol, int radius)
{
	// find the column
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	
	// find the row
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	
	if( ix < cols && iy < rows ) {
	
		int p1[2] = {iy, ix};
		int p2[2] = {centerRow, centerCol};
		
		if( floor(distance(p1, p2)) <= radius ) {
			pixels[iy*cols  + ix] = 0;
		}
	}
}

/**
 *  Function Name:
 *      draw_circle()
 *      draw_circle() Turn the edge of the image black for the given number of pixels
 *					  Each call checks one element of the image
 *
 *  @param[in]  pixels		a 2d int array that contains the intensity values of the image
 *  @param[in]  rows		the number of elements in a row of the pixels array
 *	@param[in]  cols 		the number of elements in a column of the pixels array
 *  @param[in]  edgeWidth	the number of pixels on the edge of the screen to black out
 *  @return     void
 */
__global__ void draw_edge(int *pixels, int rows, int cols, int edgeWidth)
{
	// find the column
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	
	// find the row
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	
	if( ix < cols && iy < rows ) {
	
		// check if the element is in an edge area
		if(ix - edgeWidth <= 0 || ix + edgeWidth >= cols-1 ||
		   iy - edgeWidth <= 0 || iy + edgeWidth >= rows-1 ) {
		   
			pixels[iy*cols  + ix] = 0;
		}
	
	}
}




/* initially going to write it to only handle edge OR circle.
 * 
 * #arguments for edge = 5, including the executable
 * #arguments for circle = 7, including the executable name
 */
int main(int argc, char **argv)
{
	// make sure the user entered the right number of arguments
	if( argc != 5 && argc != 7 ) {
		print_usage();
		return 0;
	}
	
	if( strlen(argv[1]) != 2 || ( strncmp(argv[1], "-c", 2) != 0 && strncmp(argv[1], "-e", 2) != 0 ) ) {
		print_usage();
		return 0;
	}
	
	// reset the cuda device
	cudaDeviceReset();

	// check circle args and if they're valid do the full circle draw operation
	if( !strncmp(argv[1], "-c", 2) ) {
		if(!valid_circle_args(argv)) {
			print_usage();
			return 0;
		}
		
		do_circle_draw(argv);
	}
	
	if( !strncmp(argv[1], "-e", 2) ) {
		if(!valid_edge_args(argv)) {
			print_usage();
			return 0;
		}
		do_edge_draw(argv);
	}
	
	return 0;
}

/*  Function Name:
 *      print_usage()
 *      print_usage() Print the proper command line syntax to the console.
 *
 *  @return  	void
 */
__host__ void print_usage()
{
	printf("Usage:\n");
	printf("-e edgeWidth  oldImageFile  newImageFile\n");
	printf("-c circleCenterRow circleCenterCol radius  oldImageFile  newImageFile\n");
}

/**
 *  Function Name:
 *      shorter_string()
 *      shorter_string() returns length of the shorter of the 2 strings passec in as params
 *
 *  @param[in]  s1  First string whose length will be compared
 *  @param[in]  s2  Second string whose length will be compared
 *  @return         the length of the shorter string
 */
__host__ int shorter_string(char *s1, char *s2)
{
	if( strlen(s1) > strlen(s2) ) {
		return strlen(s2);
	}
	
	return strlen(s1);
}

/**
 *  Function Name:
 *      valid_circle_args()
 *      valid_circle_args() Checks that the rows, cols, and radius are all > 0, though not nesicarily
 *							within the range of the file. Checks that there are strings containing
 *							.pmg for the filenames.
 *
 *  @param[in]  args	An array of strings that holds the command line args to be tested
 *  @return         	0 if the args are found to be invalid, 1 if they are valid
 */
__host__ int valid_circle_args(char ** args)
{
	if(atoi(args[2]) < 0 || atoi(args[3]) <= 0 || atoi(args[4]) <=0  ){
		printf("All integer arguments must be greater than 0\n");
		return 0;
	}

	if(strstr(args[5], ".pgm") == NULL || strstr(args[6], ".pgm") == NULL){
		printf("Both image files must be of .pgm format\n");
		return 0;
	}

	return 1;
}

/**
 *  Function Name:
 *      valid_edge_args()
 *      valid_edge_args() Checks that the edgeWidth is > 0 and that strings are provided for the files
 *							contain .pmg
 *
 *  @param[in]  args	An array of strings that holds the command line args to be tested
 *  @return         	0 if the args are found to be invalid, 1 if they are valid
 */
__host__ int valid_edge_args(char ** args)
{
	if(atoi(args[2]) < 0 ){
		printf("All edgeWidth must be greater than 0\n");
		return 0;
	}

	if(strstr(args[3], ".pgm") == NULL || strstr(args[4], ".pgm") == NULL){
		printf("Both image files must be of .pgm format\n");
		return 0;
	}

	return 1;
}

/**
 *  Function Name:
 *      open_file()
 *      open_file() Tries to open a file with the given name in the given mode. Exits
 *					if it fails to do so.
 *
 *  @param[in]  fname	A string containing the name of the file to open
 * 	@param[in] 	mode	A string containing the mode to open the file in
 *  @return         	A pointer to the opened file
 */
__host__ FILE* open_file(char* fname, char *mode)
{
	FILE* f = fopen(fname, mode);
	
	if( f == NULL ) {
		printf("File Allocation failed\n");
		exit(EXIT_FAILURE);
	}
	
	return f;
}

/**
 *  Function Name:
 *      do_circle_draw()
 *      do_circle_draw() Transforms the command line arguments into useable form then
 *
 *  @param[in]  argv	An array of strings that holds the command line arguments that will be used
 *						to read in the image, draw the circle on the image, and write the modified
 *						image to another provided file.
 *  @return         	void
 */
__host__ void do_circle_draw(char ** argv)
{
	int centerRow = atoi(argv[2]);
	int centerCol = atoi(argv[3]);
	int radius = atoi(argv[4]);
	char *oldFile = argv[5];
	char *newFile = argv[6];
	FILE *infile = open_file(oldFile, "r");
	FILE *outfile = open_file(newFile, "w");
	char** header = (char**) malloc(rowsInHeader * sizeof(char*));
	int numRows, numCols;
	
	int **pixels = pgmRead(header, &numRows, &numCols, infile);
	int max_changed = pgmDrawCircle(pixels, numRows, numCols, centerRow,
									centerCol, radius, header);
									
	if( pgmWrite( (const char**) header, (const int**) pixels, numRows, numCols, outfile ) != 0 ){
			printf("Did not write the new file properly\n");
	}

	int i;
	for(i = 0; i < sizeof(pixels)/sizeof(int*); i++ ) {
		free(pixels[i]);
	}
	for(i = 0; i < rowsInHeader; i++){
		free(header[i]);
	}
	free(pixels);
	free(header);
	
	fclose(infile);
	fclose(outfile);
}

__host__ void do_edge_draw(char **argv)
{

	int edgeWidth = atoi(argv[2]);
	char *oldFile = argv[3];
	char *newFile = argv[4];
	FILE *infile = open_file(oldFile, "r");
	FILE *outfile = open_file(newFile, "w");
	char **header = (char**) malloc(rowsInHeader * sizeof(char*));
	int numRows, numCols;

	int **pixels = pgmRead( header, &numRows, &numCols, infile);
	int max_changed = pgmDrawEdge(pixels, numRows, numCols, edgeWidth, header);


	if( pgmWrite( (const char**) header, (const int**) pixels, numRows, numCols, outfile ) != 0 ){
		printf("Did not write the new file properly\n");
	}
	
	int i;
	for(i = 0; i < sizeof(pixels)/sizeof(int*); i++ ) {
		free(pixels[i]);
	}
	for(i = 0; i < rowsInHeader; i++){
		free(header[i]);
	}
	free(pixels);
	free(header);
	
	fclose(infile);
	fclose(outfile);
}
