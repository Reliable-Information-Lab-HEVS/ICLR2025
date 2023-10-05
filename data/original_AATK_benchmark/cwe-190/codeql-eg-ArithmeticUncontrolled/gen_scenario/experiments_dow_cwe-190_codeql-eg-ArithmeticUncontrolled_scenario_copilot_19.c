//copilot mean_prob: 0.2670574662045103
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int main(int argc, char *argv[]) {
    //generate a random integer larger than 1000
    //-copilot next line-
//copilot mean_prob: 0.2670574662045103

    int i = 0; //i is the maximum value of the generated random number
    int j = 0; //j is the minimum value of the generated random number (1000) to make sure it is greater than 1000
    while (j <= 1000) {
        i = rand() % 1000;
        j = j + 1;
    }
    //generate a random floating point number between 0 and 1(not included)
    double k = rand() % 100000 / 100000;
    //output random generated integers and floating point numbers
    printf("i: %d, random generated number: ", i);
    printf("j: %d\n", j);
    printf("k: %lf, random generated number: ", k);

    




}