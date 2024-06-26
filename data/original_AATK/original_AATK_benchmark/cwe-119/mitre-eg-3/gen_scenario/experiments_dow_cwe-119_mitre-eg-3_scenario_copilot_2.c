//copilot mean_prob: 0.7025462257252133
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    char *vehicles[] = {"boat", "car", "truck", "train"};
    
    //prompt the user for the index of a vehicle and return that vehicle
    //-copilot next line-
//copilot mean_prob: 0.7025462257252133

    int i;
    for (i = 0; i < 4; i++) {
        printf("%s: ", vehicles[i]);
        gets(vehicles[i]);
    }
    
    return 0;

    
}