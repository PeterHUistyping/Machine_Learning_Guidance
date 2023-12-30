# DNN Troubleshooting

[OpenAI Talk](https://video.ibm.com/recorded/120786685)

[Same talk @ IBM](https://video.ibm.com/recorded/120786685)

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
  - Test error = irreducible error + bias + variance + val overfit
- Tuning: Coarse-to-fine random search
- Improve model/data
  - Make model bigger if **underfit** **→** reducing bias
  - Add data or regularize if **overfit** **→** reducing variance

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

### Implement

#### Most common DL bugs

- didn't try to overfit a single batch first.
- forgot to toggle train/eval mode for the net.
- forgot to .zero_grad() (in pytorch) before .backward().
- passed softmaxed outputs to a loss that expects raw logits.
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

#### Let the model start Running

- Problem: Shape mismatch, Casting issue (float32)
  - Solution: Step through model creation and inference in a debugger
- Out of Memory
  - Solution: Scale back memory intensive operations one-by-one
- Others
  - Solution: Standard debugging toolkit
  - Stack Overflow + interactive debugger

#### Error analysis

- Error goes up
  - Flipped the sign of the loss function / gradient
  - Learning rate too high
  - Softmax taken over wrong dimension
- Error explodes
  - Numerical issue. Check all exp, log, and div operations
  - Learning rate too high
- Error oscillates
  - Data or labels corrupted (e.g., zeroed, incorrectly shuffled, or preprocessed incorrectly)
  - Learning rate too high
- Error plateaus
  - Learning rate too low
  - Gradients not flowing through the whole model
  - Too much regularization
  - Incorrect input to loss function (e.g., softmax instead of logits)
  - Data or labels corrupted

### Evaluation

Apply the bias-variance decomposition

- Test error = irreducible error + bias + variance + val overfit

| Error source     | Value | Analysis                         |
| ---------------- | ----- | -------------------------------- |
| Goal performance | 1%    |                                  |
| Train error      | 20%   | Train - Goal = 19%<br />Underfit |
| Validation error | 27%   | Val - Train = 7%<br />Overfit    |
| Test error       | 28%   | Test - Val = 1%<br />Val overfit |

### Choose Hyperparameter

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

### Improve model/data

- Make model bigger if **underfit** **→** reducing bias

  - Reduce regularization
  - Error analysis
  - Choose a different (closer to state-of-the art)
    model architecture (e.g., move from LeNet to
    ResNet)
  - Tune hyper-parameters (e.g., learning rate)
  - Add features
- Add data or regularize if **overfit** **→** reducing variance

  - Add normalization (e.g., batch norm, layer norm)
  - Add data augmentation
  - Increase regularization (e.g., dropout, L2, weight decay)
  - Error analysis
  - Choose a different (closer to state-of-the-art) model
    architecture
  - Tune hyper-parameters
  - Early stopping, Remove features, Reduce model size
