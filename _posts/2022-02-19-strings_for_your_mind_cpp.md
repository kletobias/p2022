---
layout: distill
title: 'String Manipulation - C++'
date: 2022-02-18 00:00:00
description: 'A small program, created as an excercise to better understand string manipulation in C++'
tags: ['CPP','string','substitution','cout']
category: 'CPP'
comments: true
---

# Intro To String Manipulation in C++

The program gets three strings `str1`,`str2` and a sub-string of `str2` called
`str3`. Several manipulations are performed on the input strings, including
a substitution based on the position of characters in `str1` and the results are
printed to `stout`.

```cpp
//
// Created by Tobias on 2019-08-21.
//

#include <iostream>

using namespace std;

int main() {
    string str1 = "To be or not to be that is the question";
    string str2 = "only ";
    string str3 = str1.substr(6,12);
    str1.insert(32, str2);
    str1.replace(str1.find("to be", 0), 5, "to jump");
    str1.erase(9, 4);
    cout << str1 << endl;
    for (int i = 0; i < str3.length(); i++)
        cout << str3[i]; cout << endl;
}
```

---

**© Tobias Klein 2022 · All rights reserved**<br>
**LinkedIn: https://www.linkedin.com/in/deep-learning-mastery/**
