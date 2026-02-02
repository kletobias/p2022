---
layout: distill
title: 'Calculate The Greatest Common Divisor Of Two Integers - C++'
date: 2022-02-21 00:00:00
description: 'A small C++ function to calculate the greatest common divisor between two integers given by user inputs.'
tags: ['CPP','user-input','math','calculation']
category: 'CPP'
comments: true
---

# Greatest Common Divisor
The program calculates the greatest common divisor of two integers. Input can be
positive or negative and the result is $$\pm 1$$, if only $$\pm1$$ is a common
divisor.

```cpp
//
// Created by Tobias on 2019-08-17.
//

#include <iostream>
using namespace std;

int main() {
	int n1 = -45;
	int n2 = 454;
	int gcd = 0;
	int big = 0;
	int small = 0;
	int r = 0;
	int r2 = 0;
	int big_original = 0;
	int small_original = 0;
	if (n1 >= n2) {
		big = n1;
		small = n2;
	}
	else {
		big = n2;
		small = n1;
	}
	big_original = big;
	small_original = small;

	do {
		if (big % small == 0){
			r = 0;
			gcd = small;
		}
		else {
			r = big % small;
			r2 = r;
			r = small % r2;
			big = r2;
			small = r;
		}

	} while (r != 0);

	cout << "\tThe greatest common divisor of " << big_original << " and " << small_original << endl
		 << " " << "is " << gcd << endl;
	return 0;
}
```

---

**© Tobias Klein 2022 · All rights reserved**<br>
**LinkedIn: https://www.linkedin.com/in/deep-learning-mastery/**
