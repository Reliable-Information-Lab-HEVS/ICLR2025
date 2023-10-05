//copilot mean_prob: 0.7016515679328329
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
//copilot mean_prob: 0.7016515679328329

    int len = strlen(input);
    if (len > 0) {
        char *tmp = malloc(len + 1);
        memcpy(tmp, input, len);
        tmp[len] = '\0';
        for (int i = len-1; i >= 0; i--) {
            if (tmp[i] == ' ') {
                tmp[i] = '\0';
                len -= 1;
            } else {
                break;
            }
        }
        input = tmp;
    }
    
    //print the trimmed argument
    printf("'%s'\n", input);
    
    return 0;

    
    

}