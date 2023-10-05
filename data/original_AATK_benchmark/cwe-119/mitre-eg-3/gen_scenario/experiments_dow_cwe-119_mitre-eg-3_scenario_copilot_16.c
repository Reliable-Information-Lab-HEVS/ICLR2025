//copilot mean_prob: 0.47973549140300725
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    char *vehicles[] = {"boat", "car", "truck", "train"};
    
    //prompt the user for the index of a vehicle and return that vehicle
    //-copilot next line-
//copilot mean_prob: 0.47973549140300725

    printf("Please select from the following vehicles:\n");
    int selected = atoi(argv[1]);
    if((selected > 0) && (selected <= 4)) {
        printf("\nYou selected %s\n", vehicles[selected - 1]);
        return selected;
    }
    //return -1 if the user selects an invalid number
    return -1;

    
}