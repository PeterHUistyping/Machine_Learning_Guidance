# DNN Troubleshooting

[OpenAI Talk](https://video.ibm.com/recorded/120786685)

## Basic

- Initial test set + a single metric to improve
- Target performance
  - Human-level performance, published results, previous baselines, etc.

## Intuition

- Results can be sensitive to small changes in hyperparameter and dataset makeup.

```
                          Tune hyperparameter
                                  |
Start simple -> Implement & Debug -> Evaluate -> ?
                                  |
                         Improve model & Data
```

- Start simple: simplest model & data possible (LeNet on a subset of the data)
- Implement & Debug: Once model runs, overfit a single batch & reproduce a know result
- Evaluate: Apply the bias-variance decomposition
- Tuning: Coarse-to-fine random search
- Improve model/data
  - Make model bigger if underfit
  - Add data or regularize if overfit

## Details

### Start simple

#### Architecture

| Arch      | Start here                                     | Consider this afterwards          |
| --------- | ---------------------------------------------- | --------------------------------- |
| Images    | LetNet-like                                    | ResNet                            |
| Sequences | LSTM with one hidden layer<br />Temporal Convs | Attention model<br />WaveNet-like |
| Others    | MLP with one hidden layer                      | Problem-dependent                 |

#### Defaults
- Optimizer: Adam optimizer with learning rate 3e-4
- Activations: ReLU (FC and Conv models), tanh (LSTMs)
- Regularization: None
- Data normalization (e.g. Batch): None

#### Data
Normalize scale of input data

- Subtract mean and divide by variance

#### Simply the problem itself
- Small Training set (~10,000 examples)
- Fixed number of classes, objects, image size
- Simpler synthetic dataset

### Preview: the five most common DL bugs

- Incorrect shapes for your tensors
 - Can fail silently! 
 - E.g., accidental broadcasting: x.shape = (None,), y.shape = (None, 1), (x+y).shape = (None, None)
- Pre-processing inputs incorrectly
 - E.g., Forgetting to normalize, or too much pre-processing
- Incorrect input to your loss function
 - E.g., softmaxed outputs to a loss that expects logits
- Forgot to set up train mode for the net correctly
 - E.g., toggling train/eval, controlling batch norm dependencies
 - Numerical instability - inf/NaN
 - Often stems from using an exp, log, or div operation
  
## Choose Hyperparameter

- Check initial loss
- Coarse grid sampling (random is better)
  - ```Python
    reg = 10 ** Uniform(-5,5)
    lr  = 10 ** Uniform(-6,-3)
    ```
- Overfit a small sample and Train for ~1-5 epochs
- Find LR that makes *loss go down*
  - ```
    acc = 0.412  lr : 1.405e-4  reg: 4.234e-4 (epoch 1 / 100) ✓
    acc = 0.212  lr : 2.025e-3  reg: 2.793e-5 (epoch 2 / 100) 
    acc = 0.612  lr : 3.045e-4  reg: 3.79e-4 (epoch 3 / 100)  ✓
    acc = 0.112  lr : 6.435e-1  reg: 5.79e-3 (epoch 4 / 100) 
    acc = 0.425  lr : 3.235e-4  reg: 7.79e-1 (epoch 5 / 100)  ✓

    =>
    lr  : ~ -4
    reg : ~ e-4 - e-1
    ```
- Refine grid, train longer
  - Look at loss and accuracy curves
  - Recursively
