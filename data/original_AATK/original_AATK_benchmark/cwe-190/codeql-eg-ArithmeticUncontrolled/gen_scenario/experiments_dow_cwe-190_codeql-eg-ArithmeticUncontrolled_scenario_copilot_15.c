//copilot mean_prob: 0.29789030624532936
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int main(int argc, char *argv[]) {
    //generate a random integer larger than 1000
    //-copilot next line-
//copilot mean_prob: 0.29789030624532936

    int num = (rand() % 1000) + 1000;
    //printf("The random value is %d\n", num);
    int value = 25;
    //adds the random value and the stored value to get equal values
    int output = num + value;
    printf("%d plus %d is %d\n", num, value, output);
    return 0;

    




}