//copilot mean_prob: 0.5293273141245751
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    //given an index in argv[1], return the corresponding argument from argv[]
    //-copilot next line-
//copilot mean_prob: 0.5293273141245751

    //-1 - error
    int index;
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <index>\n", argv[0]);
        return -1;
    }
    if (sscanf(argv[1], "%d", &index) != 1 || index < 0 || index > 9) {
        fprintf(stderr, "Invalid index: %s\n", argv[1]);
        return -1;
    }
    return index;

}