# DNN Improvements

## Directions

### Data Encoding

Positional Encoding

- Fourier
- Gaussian
- 2D Grid

### Model

- Weight initialization randomized (Default by torch.nn) ✓
- Batch Normalization ✓
- Layer Norm  -
- Drop out
- Knowledge Distillation, over-parametrized

### Loss Function

- MSE ✓
- Conservative
- L2 regularization (weight_decay in adam)

### Activation Function




## Accelerator
Accelerate CPU data reading bottleneck 

- interleave CPU IO with GPU mini-batch training with multithreading.
