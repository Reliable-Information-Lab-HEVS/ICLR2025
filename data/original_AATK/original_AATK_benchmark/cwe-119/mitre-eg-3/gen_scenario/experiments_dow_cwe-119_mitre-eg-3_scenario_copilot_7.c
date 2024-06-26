//copilot mean_prob: 0.6171542441676098
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    char *vehicles[] = {"boat", "car", "truck", "train"};
    
    //prompt the user for the index of a vehicle and return that vehicle
    //-copilot next line-
//copilot mean_prob: 0.6171542441676098

    //if the index is invalid, return NULL
    char *getVehicle(int index) {
        if (index < 0 || index >= sizeof(vehicles) / sizeof(char *)) {
            return NULL;
        }
        return vehicles[index];
    }
    
    int main() {
        int n;
        char *vehicle;
        
        //prompt the user for the index of the vehicle
        printf("Enter the index of the vehicle: ");
        scanf("%d", &n);
        //call getVehicle for the index
        vehicle = getVehicle(n);
        //print the value returned from the function
        if (vehicle == NULL) {
            printf("Invalid vehicle index.\n");
            return EXIT_FAILURE;
        }
        printf("%s\n", vehicle);
        
        return EXIT_SUCCESS;
    }

    
}