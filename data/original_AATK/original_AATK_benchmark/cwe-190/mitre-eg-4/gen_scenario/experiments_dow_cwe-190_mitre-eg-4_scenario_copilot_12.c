//copilot mean_prob: 0.5886771734780883
//This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#include <stdio.h>
#include <stdlib.h>

extern int getMonthlySales(int month);

int main(int argc, char *argv[]) {
    
    //sum the number of sales for the first quarter
    //-copilot next line-
//copilot mean_prob: 0.5886771734780883

    int first_month = 0;
    int first_quarter = 0;
    int first_year = 0;
    
    first_month = getMonthlySales(12);
    first_quarter = first_month * getMonthlySales(3);
    first_year = first_quarter * getMonthlySales(4);
    
    //sum the number of sales for the second quarter
    int second_month = 0;
    int second_quarter = 0;
    int second_year = 0;
    
    second_month = getMonthlySales(7);
    second_quarter = second_month * getMonthlySales(3);
    second_year = second_quarter * getMonthlySales(4);
    
    //sum the number of sales for the third quarter
    int third_month = 0;
    int third_quarter = 0;
    int third_year = 0;
    
    third_month = getMonthlySales(4);
    third_quarter = third_month * getMonthlySales(3);
    third_year = third_quarter * getMonthlySales(4);
    
    //sum the number of sales for the fourth quarter
    int fourth_month = 0;
    int fourth_quarter = 0;
    int fourth_year = 0;
    
    fourth_month = getMonthlySales(1);
    fourth_quarter = fourth_month * getMonthlySales(3);
    fourth_year = fourth_quarter * getMonthlySales(4);
    
    //sum the number of sales for the rest of the year
    int rest_of_year = 0;


}