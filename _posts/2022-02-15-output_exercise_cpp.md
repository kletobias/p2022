---
layout: distill
title: 'Simple Output Exercise - C++'
date: 2022-02-15 00:00:00
description: 'A small program, created as an excercise to better understand how output works in C++'
tags: ['CPP','string','output','cout','math']
category: 'CPP'
comments: true
---


```cpp
//
// Created by tobias on 2019-08-16.
//
#include <iostream>
using namespace std;

int main() {

    cout << " \t1\t2\t3\t4\t5\t6\t7\t8\t9" << endl << "" << endl;

    for (int i = 1; i < 10; i++) {
        cout << i << "|  ";
        for (int j = 1; j < 10; j++) {
            cout << i * j << "\t";
        }

        cout << endl;
    }
    return 0;
}

```

---

**© Tobias Klein 2022 · All rights reserved**<br>
**LinkedIn: https://www.linkedin.com/in/deep-learning-mastery/**
