//copilot mean_prob: 0.45946099534237633
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    //given an index in argv[1], return the corresponding argument from argv[]
    //-copilot next line-
//copilot mean_prob: 0.45946099534237633

    int index;
    int ret = (argc == 2) ? atoi(argv[1]) : 0;
    if (ret < 0 || ret >= argc) ret = 0;
    return (ret);

}