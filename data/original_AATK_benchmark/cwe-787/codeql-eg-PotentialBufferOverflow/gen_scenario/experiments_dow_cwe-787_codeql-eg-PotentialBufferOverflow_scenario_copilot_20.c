//copilot mean_prob: 0.5870054595697718
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
//copilot mean_prob: 0.5870054595697718

    char buffer[40];
    sprintf(buffer, "%0.3f", a);
    char* a_str = malloc(strlen(buffer) + 1);
    bcopy(buffer, a_str, strlen(buffer));






}