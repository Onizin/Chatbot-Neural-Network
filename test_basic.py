import torch
import torch.nn as nn
import numpy as np
import json

# Test simple neural network
print("Testing PyTorch basic operations...")

# Test 1: Simple tensor operations
x = torch.randn(1, 10)
print("Test 1 - Tensor creation:", x.shape)

# Test 2: Simple linear layer
linear = nn.Linear(10, 5)
out = linear(x)
print("Test 2 - Linear layer:", out.shape)

# Test 3: Test with ReLU
relu = nn.ReLU()
out = relu(out)
print("Test 3 - ReLU activation:", out.shape)

# Test 4: Test softmax
softmax = nn.Softmax(dim=1)
probs = softmax(out)
print("Test 4 - Softmax:", probs.shape, "Sum:", probs.sum().item())

print("All basic PyTorch operations work!")

# Test 5: Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)
print("Test 5 - Intents loaded, count:", len(intents['intents']))

print("Basic tests completed successfully!")
