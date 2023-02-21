---
layout: distill
title: 'Calculate And Compare Different Compensation Schemes - C++'
date: 2022-02-20 00:00:00
description: 'Calculate the weekly pay comparing three different compensation schemes depending on user input - C++'
tags: ['CPP','user-input','math','calculation','optimization']
category: 'CPP'
comments: true
---
```cpp
//
// Created by Tobias on 2019-08-21.
//
#include <iostream>
using namespace std;

#define shoe_price 225
#define comp1_weekly_wage 600
#define comp2_hourly_wage 7.0
#define comp2_commission 0.1
#define comp3_commission 0.2
#define comp3_bonus_per_unit 20

// A function to get the weekly sales of units
int GetInput() {
    int units;
    cout << "Enter a number for the weekly sales: " << flush;
    if (!(cin >> units)) {
        cout << "Units in integer number only BUDDY" << endl;
        cin.clear();
        cin.ignore(1000,'\n');
        return 0;
    }
    else {
        cout << "The number is: " << units << endl;
        return units;
    }
}
// Method 1
void CalcMethod1() {

    cout << "Weekly wage is: " << comp1_weekly_wage << endl;
}

// Method 2
void CalcMethod2(int sales) {
    double per_hour = comp2_hourly_wage;
    double week_hours = 40;
    double fixed_comp2 = week_hours * per_hour;
    double variable_comp2 = sales * shoe_price * comp2_commission;
    double result_comp2 = fixed_comp2 + variable_comp2;
    cout << "Weekly wage using compensation scheme 2 is: " << result_comp2 << endl;
}
// Method 3
void CalcMethod3(int sales) {
    double result_comp3 = sales * (comp3_bonus_per_unit + comp3_commission * shoe_price);
    cout << "Weekly wage using compensation scheme 3 is: " << result_comp3 << endl;
}



int main() {
    int WeeklySales;
    WeeklySales = GetInput();
    if (WeeklySales <= 0) {
        return 0;
    }
    CalcMethod1();
    CalcMethod2(WeeklySales);
    CalcMethod3(WeeklySales);

}
```
