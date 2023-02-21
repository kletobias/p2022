---
layout: distill
title: 'A Guessing Game Written In C++'
date: 2022-02-16 00:00:00
description: 'A guessing game on the command line using user input.'
tags: ['CPP','user-input','game','cout']
category: 'CPP'
comments: true
---

# Guess The Correct Integer Between 1-100
The user must guess what the integer is, that the program picked beforehand. For
every wrong guess, the user gets a hint saying if the input was too high or too
low until the correct number is guessed.

```cpp
//
// Created by tobias on 2019-08-16.
//

#include <iostream>
#include <time.h>
#include <cstdlib>
using namespace std;


int main() {
    int input_var = 0;
	// Initialize random seed
	srand (time(NULL));
	int x = rand() % 100 + 1;
    do {
		cout << "" << endl
			 << "\tGuess which integer I picked in the interval [1,100]." << endl
             << "\t(-1 to exit) and try non integer for fun :)" << endl
             << "" << endl
             << "\tMay RNGesus be with you." << endl
             << "" << endl
             << "\tPlease enter the number now: " << flush;
        if (!(cin >> input_var)) {
        	cout << "" << endl
				 << "You entered a non-numeric entry." << endl;
			cin.clear();
			cin.ignore(1000, '\n');
			cout << "You can try again" << endl
				 << "" << endl;
		}
        if (input_var > x) {
			cout << "" << endl
				 << "Your guess was too high, try a number lower than: " << input_var << endl;
        }
        if ((input_var < x) & (input_var != -1)) {
			cout << "" << endl
				 << "Your guess was too low, try a number higher than: " << input_var << endl;
        }
    } while (input_var != (x|-1));
    if (input_var == x) {

		cout << "" << endl
			 << "You so GUCCI, you actually guessed the right number! Congrats! Exiting now. Bye" << endl;
    }
    else {
		cout << "" << endl
			 << "Exiting. Bye for now." << endl;
    }

    return 0;
    }
```
