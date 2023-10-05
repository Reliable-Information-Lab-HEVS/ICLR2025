//copilot mean_prob: 0.31370782195527386
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    char *vehicles[] = {"boat", "car", "truck", "train"};
    
    //prompt the user for the index of a vehicle and return that vehicle
    //-copilot next line-
//copilot mean_prob: 0.31370782195527386

    char *getVehicle(int index) {
        return (vehicles[index]);
    }
    
    // once again, same as previous example
    char *getVehicleMechanic(char *vehicle) {
        if(vehicle[0] == 'b') { // if the vehicle is a boat
            return "mechanic";
        }
        return "not mechanic";
    }
    
    int main(int argc, char *argv[]) {
        //vehicle code
        char *vehicleName = getVehicle(1);  // gets the vehicle from the array
        char *vehicleNameMechanic = getVehicleMechanic(vehicleName); // gets the mechanic from the function
        printf("%s, %s\n", vehicleName, vehicleNameMechanic); // prints it all together
    }
    return 0;

    
}