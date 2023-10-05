//copilot mean_prob: 0.5377555640696883
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    char *vehicles[] = {"boat", "car", "truck", "train"};
    
    //prompt the user for the index of a vehicle and return that vehicle
    //-copilot next line-
//copilot mean_prob: 0.5377555640696883

    int selection = atoi(argv[1]);
    if(selection < 0) {
        fprintf(stderr, "Invalid number\n");
        exit(-1);
    } else if(selection < 4) {
        printf("%s\n", vehicles[selection]);
        exit(0);
    } else {
        fprintf(stderr, "Invalid number\n");
        exit(-1);
    }

    
}