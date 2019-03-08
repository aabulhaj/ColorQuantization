# Image Color Quantization
Python implementation of [quantization](https://en.wikipedia.org/wiki/Color_quantization).

Assume that the following is a histogram of some image:
<p align="center"><img src="equations/illustration.png"></p>

Z values are initialized such that the number of pixels between z<sub>i+1</sub> and z<sub>i</sub> is approximately the same (for every i). Notice that we initalize k + 1 values where k is the number of the wanted quantizations.


We will use SSD (sum of squared differences) as quantization error.
Although SSD is a poor error measure compared to human perception, it is the simplest to implement.

Therefore, our goal is to minimize the following equation:

<p align="center"><img src="equations/min_goal.png" height="100"></p>


After solving the equation we get:
<p align="center"><img src="equations/zi.png" height="100"></p>


<p align="center"><img src="equations/qi.png" height="100"></p>


# Example

Original image:
<p align="center"><img src="images/gray_orig.png"></p>

Quantization to 3 colors:
<p align="center"><img src="output/quant_3.png"></p>

Quantization to 7 colors:
<p align="center"><img src="output/quant_7.png"></p>
