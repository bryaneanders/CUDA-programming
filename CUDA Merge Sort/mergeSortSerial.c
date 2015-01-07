// grabbed this from http://www.cquestions.com/2011/07/merge-sort-program-in-c.html
// modified it for this

#include <stdio.h>
#include <stdlib.h>

#include "mergeSortSerial.h"

void partition(unsigned int **keys, unsigned int **vals, unsigned int low,unsigned int high){

    unsigned int mid;

    if(low<high){
         mid=(low+high)/2;
         partition(keys, vals,low,mid);
         partition(keys, vals,mid+1,high);
         mergeSort(keys, vals,low,mid,high, high-low+1);
		// printf("low = %u, high = %u\n", low, high);
    }
}

void mergeSort(unsigned int **keys, unsigned int **vals, unsigned int low,unsigned int mid,unsigned int high, unsigned int size){

    unsigned int i,m,k,l;
	unsigned int *keysTemp = (unsigned int*) calloc(size, sizeof(unsigned int));
	unsigned int *valsTemp = (unsigned int*) calloc(size, sizeof(unsigned int));
	
	if(!keysTemp || !valsTemp){
		printf("Could not allocate temp mempory\n");
		exit(-1);
	}
		
    l=low;
    i=low;
    m=mid+1;


    while((l<=mid)&&(m<=high)){

         if((*keys)[l]<=(*keys)[m]){
             keysTemp[i-low]=(*keys)[l];
			 valsTemp[i-low]=(*vals)[l];
             l++;
         }
         else{
			keysTemp[i-low]=(*keys)[m];
			valsTemp[i-low]=(*vals)[m];
            m++;
         }
         i++;
    }
	
    if(l>mid){
         for(k=m;k<=high;k++){
             keysTemp[i-low]=(*keys)[k];
			 valsTemp[i-low]=(*vals)[k];
             i++;
         }
    }
    else{
         for(k=l;k<=mid;k++){
             keysTemp[i-low]=(*keys)[k];
			 valsTemp[i-low]=(*vals)[k];	
             i++;
         }
    }
   
    for(k=0;k<=high-low;k++){
         (*keys)[k+low]=keysTemp[k];
		 (*vals)[k+low]=valsTemp[k];
    }
	
	free(keysTemp);
	free(valsTemp);
}