# Simple-Neural-Network-Training-Example

This repository contains a simple neural network implementation in Python. The neural network uses a sigmoid activation function and adjusts weights through training iterations.

### Installation

No special installation is required for this project. Just make sure you have Python installed.

### Usage

To run the program, simply execute the `nn_example.py` script.

```bash
python nn_example.py
```

### Code Explanation

```python
from math import exp
from random import random as r

def s(x):
    return 1 / (1 + exp(-x))

i = [1, 0, 1]

w = []
for _ in range(3):
    w = w + [r()]

print("Random initial weights:")
print(w)

to = 1  

for _ in range(20000):
    il = i

    o = 0
    for j in range(3):
        o = o + il[j] * w[j]

    o = s(o)   

    err = to - o

    adj = [0] * 3

    for j in range(3):
        adj[j] = il[j] * (err * (o * (1 - o)))
        w[j] = w[j] + adj[j]

print("Weights after training:")
print(w)


o = 0
for j in range(3):
    o = o + i[j] * w[j]

o = s(o) 

print("Result:")
print(o)
```

### How It Works

1. **Initialization:**
   - The program begins by defining a sigmoid activation function `s`.
   - An input list `i` and an initial weight list `w` with random values are created.

2. **Training:**
   - The program trains the neural network over 20,000 iterations.
   - In each iteration, the output is calculated by summing the product of inputs and weights, then passing through the sigmoid function.
   - The error is computed as the difference between the target output (`to`) and the actual output.
   - The weights are adjusted based on the error using gradient descent.

3. **Result:**
   - After training, the adjusted weights are printed.
   - The program then computes the output using the trained weights and prints the final result.

### First Attempt

This is a simple implementation to demonstrate the basic principles of neural networks. Improvements and enhancements can be made for more complex tasks.
