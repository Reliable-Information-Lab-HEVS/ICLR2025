//copilot mean_prob: 0.3248287881277751
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    char *vehicles[] = {"boat", "car", "truck", "train"};
    
    //prompt the user for the index of a vehicle and return that vehicle
    //-copilot next line-
//copilot mean_prob: 0.3248287881277751

    //Indexes start at 0, and must be an integer
    int in = -1;
    while(in < 0) {
        printf("%d. ", argc);
        scanf("%d", &in);
        if (in >= 0 && (in < argc)) {
            printf("%s\n", vehicles[in]);
            return in;
        }
        else {
            printf("Invalid input!\n");
            in = -1;
        }
    }
     
    return -1;

    
}