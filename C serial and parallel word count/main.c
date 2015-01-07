/*
 * main.c
 * 
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "word_tokenize.h"
#include "merge.h"
#include "wordll.h"
#include "main.h"

#define NUM_THREADS 4

// main 
int main() 
{
	FILE *outfile, *file = open_file("testfile2", "r");
	double then, now, scost, pcost;
	Word* word_array;
	
	// serial code 
	then = currentTime();
	line_manager(file);
	to_array(&word_array);
	
	// sort the word list alphabetically and serially
	printf("%% Serially sorting serially tokenized words alphabetically\n"); 
	alpha_merge_sort(&word_array, 0, get_length()-1);
	rewind(file);
	
	// print the serially sorted alphabetically ordered list 
	outfile = open_file("SerialSortedWord.txt", "w");
	print_list(outfile, &word_array);
	fclose(outfile);
	
	to_array(&word_array);
	printf("%% Serially sorting serially tokenized words by count\n"); 
	// sort the word list numerically and serially 
	
	num_merge_sort(&word_array, 0, get_length()-1);
	rewind(file);
	
	// print the serially sorted numerically ordered list 
	outfile = open_file("SerialSortedCount.txt", "w");
	print_list(outfile, &word_array);
	fclose(outfile);
	
	// calculate the time it to the serial code to run
	now = currentTime();
	scost = now - then;
	printf("%% Serial Code executed in %lf seconds\n", scost);

	to_array(&word_array);
	
	// start of paralellel alpha sorting code
	then = currentTime();
	
	// parallel alphabetical merge sort
	pline_manager(file);
	printf("%% Parallely sorting parallely tokenized words alphabetically\n"); 
	alpha_pmerge_sort(&word_array);
	rewind(file);
	
	// print the parallelly sorted numerically ordered list
	outfile = open_file("ParallelSortedWord.txt", "w");
	print_list(outfile, &word_array);
	fclose(outfile);
	
	// parallel numerical merge sort
	pline_manager(file);
	printf("%% Parallely sorting parallely tokenized words by count\n"); 	
	num_pmerge_sort(&word_array);
	
	// print the parallelly sorted alphabetically ordered list
	outfile = open_file("ParallelSortedCount.txt", "w");
	print_list(outfile, &word_array);
	fclose(outfile);

	now = currentTime();
	pcost = now - then;
	printf("%% Parallel Code executed in %lf seconds\n", pcost);
	
	printf("%% Speed up (serial time / parallel time) with %d threads is %lf\n", NUM_THREADS, scost/pcost);
	printf("%% Efficiency gain (speed up/cores) with %d threads is %lf\n", NUM_THREADS, scost/pcost/NUM_THREADS);
	
	// close the file
	fclose(file);
	
	reset_list();
	free(word_array);
	
	return 0;
} // end main 

// returns the current time
// from our labs
double currentTime()
{

   struct timeval now;
   gettimeofday(&now, NULL);
   
   return now.tv_sec + now.tv_usec/1000000.0;
} // end currentTime
