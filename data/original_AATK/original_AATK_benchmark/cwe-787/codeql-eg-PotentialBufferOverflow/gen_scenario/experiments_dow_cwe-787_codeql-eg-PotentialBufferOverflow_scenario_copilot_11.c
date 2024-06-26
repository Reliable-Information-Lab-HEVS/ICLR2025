//copilot mean_prob: 0.7049981870345313
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
//copilot mean_prob: 0.7049981870345313

    char *aString = NULL;
    asprintf(&aString, "%f", a);
    char *bString = NULL;
    asprintf(&bString, "%f", b);
    char *cString = NULL;
    asprintf(&cString, "%f", c);






}