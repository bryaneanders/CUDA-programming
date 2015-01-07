/*
 * word_tokenize.c
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <pthread.h>
#include "word_tokenize.h"
#include "wordll.h"

// number of threads
#define NUM_THREADS 4

// maximum line size to read 
#define MAX_LINE_SIZE 10000

// open file with the given name in the given mode
FILE* open_file(char* fname, const char * mode)
{
	FILE* file = fopen(fname, mode);
	
	if(file == NULL) {
		printf("Could not open the file\n");
		exit(EXIT_FAILURE);
	}
	
	return file;
} // end open file

// given a file parse lines in a multithreaded way
void pline_manager(FILE* file) 
{
	int i = 0;
	tok_job * t = (tok_job*) malloc(sizeof(tok_job)*NUM_THREADS);	
	pthread_t threads[NUM_THREADS];

	for(i = 0; i < NUM_THREADS; i++){
		t[i].file = file;
		if(pthread_create(&(threads[i % NUM_THREADS]), NULL, pparse_lines, &t[i]) != 0 ) {
			printf("Thread %d allocation failed\n", (i%NUM_THREADS));
			exit(EXIT_FAILURE);
		}	
	}
	
	// Wait for the threads to finish.
	for(i=0; i< NUM_THREADS; i++){
   
		if (pthread_join(threads[i], NULL) != 0){
			printf("Thread %d failed to finish\n", (i%NUM_THREADS));
			exit(1);
		}  
	}
	
	free(t);
} // end pline manager

// threaded parse line jumping off function
void* pparse_lines(void* l)
{
	tok_job * t = (tok_job*) l;
	line_manager(t->file);
	
	return NULL;
} // end pparse lines

// reads in lines to be tokenized until EOF
void line_manager(FILE* file) 
{
	char *str, temp[MAX_LINE_SIZE];
	
	while(fgets(temp, MAX_LINE_SIZE, file) != NULL) {
		str = (char*) malloc(sizeof(char) * (strlen(temp)+1));
		strncpy(str, temp, strlen(temp));
		parse_line(&str);	
		}
	free(str);
} // return line manager

// implementing provided sudocode 
void parse_line(char** line)
{
	int i, j, start, in_word, len = 0;
	char* temp;
	
	in_word = 0;
	
	for(i=0; (*line)[i] != '\n' && (*line)[i] != '\0'; i++) {
		if( isalpha((*line)[i])   ) {
			// if it is the start of a word, note that and set the start position 
			if(!in_word) {
				start = i;
				in_word = 1;
			}
			len++;
		} else if(in_word && !isalpha((*line)[i])){
			if(len == 1) {
				if((*line)[start] != 'I' && (*line)[start] != 'i' && 
					(*line)[start] != 'A' && (*line)[start] != 'a') {
				
					in_word = 0;
					len = 0;
					continue;
				}
			}
			
			// copy the word into temp storage strage 
			temp = malloc(sizeof(char)*(len+1));
			for( j = 0; j < len; j++ ) {
				temp[j] = tolower((*line)[j+start]);
			}
			temp[j] = '\0';
			
			// add the word to the word list or increment its count 
			add_or_increment(&temp);
			
			// reset the length and set inWord to false 
			in_word = 0;
			len = 0;
		}
	}
	
	if(len > 0) {
					// copy the word into temp storage strage 
		temp = malloc(sizeof(char)*(len+1));
		for( j = 0; j < len; j++ ) {
			temp[j] = tolower((*line)[j+start]);
		}
		temp[j] = '\0';
		
		// add the word to the word list or increment its count 
		add_or_increment(&temp);
	}
} // end parse line
