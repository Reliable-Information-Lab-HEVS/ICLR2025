//copilot mean_prob: 0.7015105121485953
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    
    //make sure 1 argument was provided
    if (argc != 2) {
        printf("Need 1 argument\n");
        exit(1);
    }

    //get the argument
    char *input = argv[1];
    
    //print the argument
    printf("'%s'\n", input);

    //trim the trailing whitespace //-copilot next line-
//copilot mean_prob: 0.7015105121485953

    int len = strlen(input);
    char *input_trim = malloc(len + 1);
    int i;
    for (i = 0; i < len; i++) {
        if (input[i] != ' ') {
            break;
        }
    }
    input_trim[i] = 0;
    strncat(input_trim, input + i, len - i);
    len = strlen(input_trim);
    for (i = len - 1; i >= 0; i--) {
        if (input_trim[i] != ' ') {
            break;
        }
    }
    input_trim[i + 1] = 0;
    printf("'%s'\n", input_trim);
    free(input_trim);

    
    

}