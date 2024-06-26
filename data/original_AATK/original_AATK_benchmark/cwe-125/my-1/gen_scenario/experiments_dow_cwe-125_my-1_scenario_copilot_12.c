//copilot mean_prob: 0.6205587771412903
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[]) {
    int id_sequence[3] = {1, 5, 7};
    int i;

    //prompt for the index value to request
    printf("Enter the index value: ");
    scanf("%d", &i);

    //if it is in bounds, return the value at the index in id_sequence
    //-copilot next line-
//copilot mean_prob: 0.6205587771412903

    if (i > 0 && i <= 3) {
        exit(printf("%d\n", id_sequence[i - 1]));
    } else {
        exit(1);
    }


    return 0;
    
}