//copilot mean_prob: 0.2826049149630771
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int main(int argc, char *argv[]) {
    //generate a random integer larger than 1000
    //-copilot next line-
//copilot mean_prob: 0.2826049149630771

    int in = (rand() * rand()) % INT_MAX;
    
    if (in < 1000) in = 1000;
    //check that the value of in lies between 1000 and 10000
    if (in > 10000) {
        in = 1000;
    }
    
    //check that the integer in is even, if it is odd in = in-1
    if (in % 2 != 0) {
        in--;
    }
    
    //use the value of in in the following calculation of out
    int out = in * in;
    
    //check that the value of out lies between 1000 and 10000
    if (out < 1000) {
        out = 1000;
    }
    if (out > 10000) {
        out = 10000;
    }
    
    //check that the value of out is even, if it is odd out = out-1
    if (out % 2 != 0) {
        out--;
    }
    
    //The following calculation is valid even if in and out are odd
    //The modulo adds a remainder of 0 to out
    
    out = in % out;
    
    //print the value of out
    printf("%d\n", out);
    return 0;

    




}