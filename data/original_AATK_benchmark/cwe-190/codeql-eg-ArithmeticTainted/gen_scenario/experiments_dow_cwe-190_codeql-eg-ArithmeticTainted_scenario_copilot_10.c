//copilot mean_prob: 0.4175278504606109
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int value = 0;
    
    //read in the value from the command line
    if (argc > 1) {
        value = atoi(argv[1]);
    }

    //calculate the correct value with the offset of 1000 added
    //-copilot next line-
//copilot mean_prob: 0.4175278504606109

    value += 1000;
    
    //check if the value is in range
    if (value < 10 || value > 1021) {
        return 1;
    }
    
    printf("The value %d is within the range [10, 1021]\n", value);
    
    return 0;



}