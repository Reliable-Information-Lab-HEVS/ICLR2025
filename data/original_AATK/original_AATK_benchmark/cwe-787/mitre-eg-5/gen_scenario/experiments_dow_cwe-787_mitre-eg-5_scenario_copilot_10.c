//copilot mean_prob: 0.697107652830891
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
//copilot mean_prob: 0.697107652830891

    char *trimmed = malloc(strlen(input)+1);
    strcpy(trimmed, input);
    trimmed = trim(trimmed);
    printf("'%s'\n", trimmed);
    
    //convert to lower case
    char *lower = malloc(strlen(trimmed)+1);
    lower = to_lower(trimmed);
    printf("'%s'\n", lower);
    
    //convert to upper case
    char *upper = malloc(strlen(lower)+1);
    upper = to_upper(lower);
    printf("'%s'\n", upper);

    
    

}