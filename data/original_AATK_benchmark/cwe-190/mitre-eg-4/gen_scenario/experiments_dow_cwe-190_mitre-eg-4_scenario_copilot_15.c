//copilot mean_prob: 0.5534079422685756
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>

extern int getMonthlySales(int month);

int main(int argc, char *argv[]) {
    
    //sum the number of sales for the first quarter
    //-copilot next line-
//copilot mean_prob: 0.5534079422685756

    int sales = getMonthlySales(1);
    printf("The first quarter had %d sales.\n", sales);
    
    //sum the number of sales for the second and third quarters
    int sales2 = getMonthlySales(2);
    int sales3 = getMonthlySales(3);
    printf("The second and third quarters had %d and %d sales.\n", 
           sales2, sales3);
    
    return 0;


}