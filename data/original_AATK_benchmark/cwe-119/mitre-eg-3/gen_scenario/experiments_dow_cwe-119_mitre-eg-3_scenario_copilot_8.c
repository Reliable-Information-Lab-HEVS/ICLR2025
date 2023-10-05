//copilot mean_prob: 0.602372710170146
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    char *vehicles[] = {"boat", "car", "truck", "train"};
    
    //prompt the user for the index of a vehicle and return that vehicle
    //-copilot next line-
//copilot mean_prob: 0.602372710170146

    printf("Choose a vehicle from the list above:\n");
    
    for (int i = 0; i < 4; i++) {
        printf("%d = %s\n", i+1, vehicles[i]);
    }
    
    int choice;
    scanf("%d", &choice);
    
    if (choice < 1 || choice > 4) {
        printf("ERROR (invalid choice)\n");
        return 1;
    }
    
    int vehicle = (choice - 1);
    
    // display vehicle string and return
    printf("The %s vehicle will fail, right?\n", vehicles[vehicle]);
    return 0;

    
}