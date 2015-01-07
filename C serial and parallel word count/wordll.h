/*
 * wordll.h
 * 
 */

#ifndef WORDLL_H
#define WORDLL_H

typedef struct Word{
	/* the number of times the word occurs */
	int count;
	/* String to hold the world */
	char *word;
	/* a pointer to the next word in the list */
	struct Word* next;
} Word;

int shorter_string_len(char* s1, char* s2);
void add_or_increment(char ** word);
void to_array(Word ** word_array);
void free_word_array(Word** a);
void reset_list();
int get_length();
void print_list(FILE* file, Word** array);
void print_array(Word ** w);
char* blank_str(int n);
Word* get_head();
void free_head();
int num_digits(int i);

#endif
