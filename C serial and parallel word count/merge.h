/*
 * merge.h
 * 
 */

#ifndef MERGE_H
#define MERGE_H

#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "word_tokenize.h"
#include "wordll.h"

void alpha_merge_sort(Word **array, int p, int r);
void alpha_merge(Word **array, int p, int q, int r);
void alpha_pmerge_sort(Word **array);
void* alpha_pmerge_thread(void* args);
void num_merge_sort(Word **array, int p, int r);
void num_merge(Word **array, int p, int q, int r);
void num_pmerge_sort(Word **array);
void* num_pmerge_thread(void* args);

#endif
