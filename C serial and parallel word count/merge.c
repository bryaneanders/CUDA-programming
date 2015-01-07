/*
 * merge.c 
 * 
 */

#include "merge.h"

#define NUM_THREADS 4

// a struct to hold one thread's job for parallel merge sorting
typedef struct JobSpec{
	// array of word objects
	Word** words;
	// start and ending points for a threaded merge sort
	int start, end;
} JobSpec;

/* both serial and parallel, alphabetical and numeric merge and merge_sort functions
 *  adapted code from http://vinayakgarg.wordpress.com/2012/11/08/merge-sort-program-in-c/ */
void alpha_merge_sort(Word **array, int p, int r) 
{
	int q; 

	if(p < r) {
		q = (p+r)/2;
	
		alpha_merge_sort(array, p, q);
		alpha_merge_sort(array, q+1, r);
		alpha_merge(array, p, q, r);
	}
} // end alpha merge sort

void alpha_merge(Word **array, int p, int q, int r) 
{
	int i, start, mid;
	Word temp[r-p];

	start = p;
	mid = q+1;
	i = 0;
	while(start <= q && mid <= r) {
		if(strncmp((*array)[start].word, (*array)[mid].word, 
			shorter_string_len((*array)[start].word, (*array)[mid].word)) < 0 )  {
			
			temp[i++] = (*array)[start++];
		} else {
			temp[i++] = (*array)[mid++];
		}
	}
	while(start <= q) {
		temp[i++] = (*array)[start++];
	}
	while(mid <= r) {
		temp[i++] = (*array)[mid++];
	}
	
	i--;
	while(i >= 0 ){
		(*array)[i+p] = temp[i];
		i--;
	} 
	
} // end alpha merge

// merge alphabetically using multiple threads 
void alpha_pmerge_sort(Word **array) 
{
	pthread_t threads[NUM_THREADS];
	JobSpec *jobs = (JobSpec*) malloc(sizeof(JobSpec)*NUM_THREADS);

	int i, mid, q1, q3;
	
	mid = get_length()/2;
	q1 = mid /2;
	q3 = 3 * q1;
	
	jobs[0].start = 0;
	jobs[0].end = q1;
	jobs[1].start = q1+1;
	jobs[1].end = mid;
	jobs[2].start = mid+1;
	jobs[2].end = q3;
	jobs[3].start = q3+1;
	jobs[3].end = get_length()-1;	
	
	// create threads which will each sort length/NTHREADS words each 
	for( i = 0; i < NUM_THREADS; i++) {
		jobs[i].words = array;
		if(pthread_create(&(threads[i]), NULL, alpha_pmerge_thread, &jobs[i]) != 0 ) {
			printf("Thread %d allocation failed\n", 0);
			exit(1);
		}
	}
	
	// Wait for the threads to finish
	for(i=0; i< NUM_THREADS; i++){
   
		if (pthread_join(threads[i], NULL) != 0){
			printf("Thread %d failed to finish\n", i);
			exit(EXIT_FAILURE);
		}  
	}

	// finish merging
	alpha_merge(array, 0, q1, mid);
	alpha_merge(array, mid+1, q3, get_length()-1);
	alpha_merge(array, 0, mid, get_length()-1);
	
	free(jobs);
} // end alpha pmerge sort

// a function that gives a single thread its job
void* alpha_pmerge_thread(void* args)
{
	JobSpec* job = (JobSpec*) args;
	alpha_merge_sort(job->words, job->start, job->end);
	
	return NULL;
} // end alpha pmerge head

//merge sort by word occurance
void num_merge_sort(Word **array, int p, int r) 
{
	int q; 
	
	if(p<r) {
		q = (p+r)/2;
	
		num_merge_sort(array, p, q);
		num_merge_sort(array, q+1, r);
		num_merge(array, p, q, r);
	}
} // end num merge sort

// merge 2 subarrays together
void num_merge(Word **array, int p, int q, int r) 
{
	int i, start, mid;
	Word temp[r-p];
	
	// printf("in num merge\n");
	i = 0;
	start = p;
	mid = q+1;
	while(start <= q && mid <= r) {
		if((*array)[start].count > (*array)[mid].count)  {
			temp[i++] = (*array)[start++];
		} else {
			temp[i++] = (*array)[mid++];
		}
	}
	while(start <= q) {
		temp[i++] = (*array)[start++];
	}
	while(mid <= r) {
		temp[i++] = (*array)[mid++];
	}
	
	i--;
	while(i >= 0 ){
		(*array)[i+p] = temp[i];
		i--;
	} 
} // end num merge

// merge by count using multiple threads
void num_pmerge_sort(Word **array) 
{
	pthread_t threads[NUM_THREADS];
	JobSpec *jobs = (JobSpec*) malloc(sizeof(JobSpec)*NUM_THREADS);

	int i, mid, q1, q3;
	
	mid = get_length()/2;
	q1 = mid /2;
	q3 = 3 * q1;
	
	jobs[0].start = 0;
	jobs[0].end = q1;
	jobs[1].start = q1+1;
	jobs[1].end = mid;
	jobs[2].start = mid+1;
	jobs[2].end = q3;
	jobs[3].start = q3+1;
	jobs[3].end = get_length()-1;	
	
	// create threads which will each sort length/NTHREADS words each 
	for( i = 0; i < NUM_THREADS; i++) {
		jobs[i].words = array;
		if(pthread_create(&(threads[i]), NULL, num_pmerge_thread, &jobs[i]) != 0 ) {
			printf("Thread %d allocation failed\n", 0);
			exit(1);
		}
	}
	
	// Wait for the threads to finish
	for(i=0; i< NUM_THREADS; i++){
   
		if (pthread_join(threads[i], NULL) != 0){
			printf("Thread %d failed to finish\n", i);
			exit(EXIT_FAILURE);
		}  
	}
	
	// finish merging 
	num_merge(array, 0, q1, mid);
	num_merge(array, mid+1, q3, get_length()-1);
	num_merge(array, 0, mid, get_length()-1);
	
	free(jobs);
}// end pmerge sort

// the threaded function that does most of the merging
void* num_pmerge_thread(void* args)
{
	JobSpec* job = (JobSpec*) args;
	num_merge_sort(job->words, job->start, job->end);
	
	return NULL;
} // end num pmerge thread
