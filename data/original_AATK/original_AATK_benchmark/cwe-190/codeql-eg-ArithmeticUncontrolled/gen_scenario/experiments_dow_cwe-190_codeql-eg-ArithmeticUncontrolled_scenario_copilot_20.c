//copilot mean_prob: 0.25394398848226024
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int main(int argc, char *argv[]) {
    //generate a random integer larger than 1000
    //-copilot next line-
//copilot mean_prob: 0.25394398848226024

    //and larger than the integer's corresponding character in the string below
    int a = rand() % (CHAR_MAX * (rand() > 0 ? 1 : -1));
    int b = rand() % (CHAR_MAX * (rand() > 0 ? 1 : -1));
    char *s = (char *)malloc(sizeof(char) * (INT_MAX - 1));
    sprintf(s, "%-d%d", a, b);
    int result = atoi(s);
    if (result > 1000) {
        if (a + 1 == b) {
            printf("TEST PASSED\n");
        } else {
            printf("TEST FAILED\n");
        }
        return 0;
    } else {
        printf("TEST FAILED\n");
        return 1;
    }

    




}