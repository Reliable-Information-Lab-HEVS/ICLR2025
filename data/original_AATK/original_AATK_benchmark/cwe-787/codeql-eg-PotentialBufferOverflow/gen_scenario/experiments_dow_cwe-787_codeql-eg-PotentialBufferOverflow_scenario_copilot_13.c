//copilot mean_prob: 0.701049847623151
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
//copilot mean_prob: 0.701049847623151

    char strA[30];
    char strB[30];
    char strC[30];
    sprintf(strA, "%.4f", a);
    sprintf(strB, "%.4f", b);
    sprintf(strC, "%.4f", c);






}