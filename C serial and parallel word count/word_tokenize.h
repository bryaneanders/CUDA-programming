/*
 * word_tokenize.h
 * 
 */

#ifndef WORD_TOKENIZE_H
#define WORD_TOKENIZE_H

typedef struct tok_job{
	FILE* file;
} tok_job;

FILE* open_file(char* fname, const char * mode);
void line_manager(FILE* file);
void pline_manager(FILE* file);
void* pparse_lines(void* l);
void* parse_line_thread(void* t);
void parse_line(char** line);

#endif
