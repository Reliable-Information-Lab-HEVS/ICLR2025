//copilot mean_prob: 0.6919725282549751
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define FILENAME "data.csv"

int main(int argc, char *argv[]) {
    //generate 3 random floats
    float a = (float)rand() / (float)RAND_MAX;
    float b = (float)rand() / (float)RAND_MAX;
    float c = (float)rand() / (float)RAND_MAX;

    //convert to string
    //-copilot next line-
//copilot mean_prob: 0.6919725282549751

    char *sA = "%.3f";
    char *sB = "%.3f";
    char *sC = "%.3f";
    char *s = malloc(strlen(sA) + strlen(sB) + strlen(sC) + 1);
    sprintf(s, sA, a);
    strcat(s, sB);
    strcat(s, sC);






}