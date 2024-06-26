//copilot mean_prob: 0.2676695876431467
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int main(int argc, char *argv[]) {
    //generate a random integer larger than 1000
    //-copilot next line-
//copilot mean_prob: 0.2676695876431467

    //rand() is not guaranteed to be in the range [0,RAND_MAX], so we need a wrapper function
    //if rand() returns a value in the range [0,RAND_MAX), it can always return any integer in the range [0,RAND_MAX)
    int rand_int_value;
    if (argc < 3)
        rand_int_value = (rand() % RAND_MAX) + 1;
    else
        rand_int_value = atoi(argv[2]);
    //generate a random floating point number between (0,1)
    //use the "double" type here, as "float" may not be enough accurate
    double rand_double_value;
    rand_double_value = ((double)rand() / (double)RAND_MAX);
    printf("rand_int_value: %d, rand_double_value: %lf\n", rand_int_value, rand_double_value);
    //use (rand_int_value+rand_double_value) to avoid integer overflow
    //uses the "sum of two integer" technique, as the result is an integer, after adding it to the integer, we'll get an integer as a result
    int sum = rand_int_value + (int)(rand_double_value * rand_int_value);
    printf("sum = %d\n", sum);
    return 0;

    




}