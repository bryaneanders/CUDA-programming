//
//  pgmUtility.h
//
//  Created by Tony Tian on 11/2/13.
//  Copyright (c) 2013 Tony Tian. All rights reserved.
//

#ifndef cscd439pgm_pgmUtility_h
#define cscd439pgm_pgmUtility_h

#include <math.h>

#define rowsInHeader 4      // number of rows in image header
#define maxSizeHeadRow 200  // maximal number characters in one row in the header




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
int ** pgmRead( char **header, int *numRows, int *numCols, FILE *in  );


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
 *                  the /\circle draw might change the maximum intensity value in the image, so we
 *                  have to change maximum intensity value in the header accordingly.
 *  @return         return 1 if max intensity is changed, otherwise return 0;
 
*/
int pgmDrawCircle( int **pixels, int numRows, int numCols, int centerRow,
                  int centerCol, int radius, char **header );

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
int pgmDrawEdge( int **pixels, int numRows, int numCols, int edgeWidth, char **header );



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
int pgmWrite( const char **header, const int **pixels, int numRows, int numCols, FILE *out );



/**
 *  Function Name:
 *		find_max_intensity()
 *		find_max_intensity() looks through a 2d array of integers, a, and returns the largest int value in the array.
 *
 *	@param[in]	a		a 2d integer array which holds the intensity values we will check
 *  @param[in]	rows	an integer which denotes the number of rows in a
 *  @param[in]	cols	an integer which denotes the number of columns in a
 *  @return				returns a default of 0, black, which is replaced by the largest, brightest, integer value in the array.
 */
int find_max_intensity(int **a, int rows, int cols);

/**
 *  Function Name:
 *		max_intensity_digits()
 *		max_intensity_digits() Determines the number of digits in a positive integer
 *
 *	@param[in]	n	The int whose length in digits will be determined	
 *
 *  @return			returns the number of digis in n if n is less than 1 million. If n is negative it returns 0.
 */
int max_intensity_digits(int n);

#endif
