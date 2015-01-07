/*
 * wordll.c
 * 
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "wordll.h"
#include "word_tokenize.h"

#define WORD_LEN 41
#define COUNT_LEN 11

Word * head = NULL;

// takes 2 strings and returns the length of the shorter one
int shorter_string_len(char* s1, char* s2)
{
	if(strlen(s1) < strlen(s2) ) { 
		return strlen(s1);
	} else {
		return strlen(s2);
	}
} // end shorter string len

// if a word i in the list, increment its count
// otherwise add it to the list
void add_or_increment(char ** w)
{
	Word *prev, *trav;
	
	if(*w == NULL) {
		printf("submitted invalid word\n");
		exit(EXIT_FAILURE);
	}
	
	// if the list is empty, create a new Word object as head
	// and store the parameter string in it
	if(head == NULL) {
		head = (Word*) malloc(sizeof(Word));
		if(head == NULL) {
			exit(EXIT_FAILURE);
		}
	
		head->next = NULL;
		head->count = 1;
		head->word = *w;
		
		return;
	}
	
	// check if the word is in the list, if it is increment 
	for(prev = trav = head; trav != NULL; prev = trav, trav = trav->next) {
		if(!strncmp(trav->word, *w, shorter_string_len(trav->word, *w))) {
			trav->count++;
			return;
		}
	}
	
	// otherwise create a new word object at the end of the list
	// and insert the parameter string into it
	trav = (Word*) malloc(sizeof(Word));
		
	if(trav == NULL ) {
		printf("Failed to add Word struct");
		exit(EXIT_FAILURE);
	}
	
	trav->next = NULL;
	prev->next = trav;	
	trav->word = *w;
	trav->count = 1;	
} // end add or increment

// convert the list into an array of word objects
void to_array(Word ** word_array)
{
	int i;
	Word *trav; 
	
	*word_array = (Word*) malloc(sizeof(Word)*get_length());
	
	if(word_array == NULL) {
		printf("cound not allocate Word* array\n");
		exit(EXIT_FAILURE);
	}
	
	// copy each element from the list to the array
	for(trav = head, i = 0; trav != NULL; trav = trav->next, i++) {
		(*word_array)[i] = *trav;
	}
} // end to array

// reset to an empty list and free the list's memory
void reset_list()
{
	Word *trav;
	
	for( trav = head->next; trav != NULL; trav = head->next ) {
		head->next = trav->next;
		free(trav->word);
	}
	
	head = NULL;
}

// return how many elements are in the list
int get_length() 
{
	Word* trav;
	int i;
	
	for( trav = head, i = 0; trav != NULL; trav = trav->next, i++ );
	
	return i; 
} // return get length

// print the file in the specified format
void print_list(FILE* file, Word** array)
{
	char* str = (char*) malloc(sizeof(char)*WORD_LEN);
	char* ct_str = (char*) malloc(sizeof(char)*COUNT_LEN);
	int i;

	if(str == NULL || ct_str == NULL) {
		printf("Failed to allocate str pr ct_str in print_string\n");
	}

	fprintf(file, "|----------------------------------------|----------|\n");
	fprintf(file, "|English Word                            |Count     |\n");
	for(i = 0; i < get_length(); i++) {
		
		// create a string that contains the word itself and a blank string
		// determined by the number of characters in the word
		sprintf(str, "%s%s", (*array)[i].word, blank_str(WORD_LEN-strlen((*array)[i].word)));
		
		// create the string that contains the count using the count itself and a blank string
		// determined by the number of digits in the count
		sprintf(ct_str, "%d%s", (*array)[i].count, blank_str(COUNT_LEN - num_digits((*array)[i].count)));
		
		fprintf(file, "|----------------------------------------|----------|\n");
		fprintf(file, "|%s|%s|\n", str, ct_str);
	}
	
	fprintf(file, "|----------------------------------------|----------|\n");
	
	free(str);
	free(ct_str);
} // end print list

// test method to print the array
void print_array(Word ** w)
{
	int i;
	for( i = 0; i < get_length(); i++ ) {
		printf("%s, %d\n", (*w)[i].word, (*w)[i].count );
	}
} // end print array

// return a string of length n with n-1 blank spaces that is null terminated
char* blank_str(int n)
{
	int i;
	char* blank;
	
	if(!(blank = (char*) malloc(sizeof(char) * n))) {
		EXIT_FAILURE;
	}
	
	for(i = 0; i <  n-1; i++) { blank[i] = ' '; }
	blank[i] = '\0';
		
	return blank;
} // end blank str

// return the head of the list
Word* get_head(){ return head; }

// free the head
void free_head(){ free(head); }

// find the number of digits in a word's count
int num_digits(int i) 
{
	if( i >= 10 ) {
		if(i >= 100) {
			if( i >= 1000) {
				if( i >= 10000 ) {
					if( i >= 100000 ) {
						if( i >= 1000000 ) {
							return 7;
						}
						return 6;
					}
					return 5;
				}
				return 4;
			}
			return 3;
		}
		return 2;
	}
	return 1;
} // end num digits
