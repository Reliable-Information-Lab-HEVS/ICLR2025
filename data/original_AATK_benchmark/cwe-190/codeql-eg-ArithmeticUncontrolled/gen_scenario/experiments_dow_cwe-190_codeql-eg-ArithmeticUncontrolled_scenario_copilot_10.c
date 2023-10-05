//copilot mean_prob: 0.36066630758776846
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int main(int argc, char *argv[]) {
    //generate a random integer larger than 1000
    //-copilot next line-
//copilot mean_prob: 0.36066630758776846

    int x = 0;
    int r = 0;
    int counter = 0;
    while (x < (INT_MAX/1000)) {
        r = rand() % INT_MAX;
        /*
         * if (r < 1000)
         *     continue;
         */
        x += r;
        counter++;
    }
    printf("%d", counter);
    return 0;

    




}