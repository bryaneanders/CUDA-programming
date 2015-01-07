#include <stdio.h>h
#include <stdin.h>
#include <string.h>
#include <regex.h>
#include "count_words.h"



int num_words;

/*chars is a signle line of text */
void count_words(char** chars, Word*** words) {
	/* just need to tokenize the content in words into Word structs,
	 * after checking if a struct for that word already exists, if it does
	 * increment the count. At this point assume words is large enough to handle
	 * all the words I throw at it */
	 
	 int i, j, start, stop, length, wordlength;
	 char * word = (char*) malloc(sizeof(char) * 75);
	 if(word == NULL) {
		 exit(1);
	 }
	 
	 length = (sizeof(chars) / sizeof(char));
	 
	 start = stop = 0;
	 for( i = 0,; i < length; i++ ) {
		/* count out the size of the word */
		if( isCharacter(i) ) { 
			 stop++; 
		} else {
			/* skip over unwanted single letters */
			if( stop-start > 1 || chars[start] == 'i' || chars[start] == 'I' ||
			chars[start] == 'a' || chars[start] == 'A' ) {
				wordlength = stop - start + 2; /* for the null character */
				/* get the word itself */
				/* this may be off by 1 */
				for( j = start; j <= stop; j++ ) {
					/* j-start to start from index 0 */
					word[j-start] = *chars[j];				
				}
				/* null terminate the string */
				word[j-start] = '\0';
				
				/* add to the words list or increment the count */
				/* not sure if i need to reference word */
				insertIntoWords(word, wordlength, words);
			}
			
			start = i + 1;
			stop = i;
		}
	 }
	free(word);
}

int isCharacter(int n) {
	if( n == ' ' || n == '\'' || n == '.' || n == ',' || n == '!' || n == '?' ||
		n == '/' || n == '>' || n == '<' || n == '@' || n == '#' || n == '$' || 
		n == '%' || n == '^' || n == '&' || n == '*' || n == '-' || n == '_' ||
		n == '[' || n == ']' || n == '{' || n == '}' || n == '(' || n == ')' || 
		n == '`' || n == '~' || n == '\\' || n == '|' || n == '"' ) 
	{ return 0;	}
	
	return 1;
}
