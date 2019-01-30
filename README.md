# GuidedFilter
Implementation of Guided Filter - Python

This code is implemented based on the MATLAB release code provided by Kaiming He[http://kaiminghe.com/]



##Before use

```
pip install numpy
```



##Python edition

When u r using **python2.x** , add the following code before running/..  -.-

```
from __future__ import division
```

**python3.x**  : No Error(s), No Running(s)



##Usage

```
import fastguidedfilter
```



##Parameters settings

```
def guideFilter(I, ra=16, eps=2, s=4):
I:   guidance Image 
r:   window redius
eps: normalize parameter
s:   sampling fraction try s = r/4 to s=r
```



##License

The MIT License (MIT) Copyright (c) 2019 [tyty](https://bravotty.github.io/)

