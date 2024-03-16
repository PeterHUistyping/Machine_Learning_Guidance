# Machine Learning Guidance

A collection of my *Machine Learning and Data Science Projects*, including both Theoretical and Practical (Applied) ML.

In addition, there are also references (paper, ebook, repo, tool, etc) that's interesting and helpful attached, ranging from beginner to advanced.

## ML Categories

### Generative vs Discriminative Model

**Generative Model**

- Joint Probability $P(x,y)$
  - Assume some functional form for $P(y)$, $P(x|y)$
  - Then, calculate $P(y|x)$ (via Bayes rule)
- Naive Bayes, HMM
  - Generate $X$ via $P(x|y)$ ✓
  - Not direct classification ✗

**Discriminative Model**

- Conditional Probability $P(y|x)$
  - Direct classification  ✓
- Logistic regression, SVM, NN

Reference: [Generative and Discriminative Model](http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf), Professor Andrew NG

### Types of Learning

- Hypothesis $h_{\omega}: R^n \rightarrow Y$, with weights $\omega$.
  - Input $x_1, ..., x_m$, where $x_i \in \R^n$
  - Labels/Output $Y=\{y_1,...,y_n\}$

| Output \ Type                                 | Unsupervised                                        | Supervised [Labels]          |
| --------------------------------------------- | --------------------------------------------------- | ---------------------------- |
| *Continuous*  <br /> $Y=\R$               | ***Clustering & <br />Dim Reduction***      | ***Regression***     |
|                                               | ○ SVD                                              | ○ Linear / Polynomial       |
|                                               | ○ PCA                                              | ○ Non-Linear Regression     |
|                                               | ○ K-means                                          | Decision Trees               |
|                                               | ○ GAN ○ VAE                                     | Random Forest                |
| *Discrete* <br /> $Y =$\{*Categories*\} | ***Association /<br /> Feature Analysis*** | ***Classification*** |
|                                               | ○ Apriori                                          | ○ Bayesian       ○ SVM |

And more,

| Aspect \ Type | Semi-Supervised  | Reinforcement               |
| ------------ | ---------------- | --------------------------- |
| Learn from   | Labels available | Rewards                     |
| Methods      | pseudo-labels    | ○ Q learning              |
|              | iteratively      | ○ Markov Decision Process |

**Reinforcement Learning**

- In a state each timestamp
  - when an action is performed, we move to a new state and receive a reward
  - No knowledge in advance of how actions affect either the new state or the reward

**Goal**

- Value-based V(s)
  - the agent is expecting a long-term return of the current states under policy π
- Policy-based
  - the action performed in every state helps you to gain maximum reward in the future
  - Deterministic: For any state, the same action is produced by the policy π
  - Stochastic: Every action has a certain probability
- Model-based
  - create a virtual model for each environment
  - the agent learns to perform in that specific environment

## Feature Engineering

- Feature Selection
  - After fitting, plot Residuals vs any Predictor Variable
  - Linearly-dependent feature vectors
- Imputation
- Handling Outliers
  - Removal, Replacing values, Capping, Discretization
- Encoding
  - Integer Encoding
  - One-Hot Encoding (enum -> binary)
- Scaling
  - Normalization, min-max/ 0-1
  - Standardization

## Applied ML Best Practice

### DNN Troubleshooting

#### Basic

- Initial test set + a single metric to improve
- Target performance
  - Human-level performance, published results, previous baselines, etc.

#### Intuition

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

[Troubleshooting](./Asset/Best%20Practice/TroubleShooting.md)

[OpenAI Talk](https://video.ibm.com/recorded/120786685)

### DNN improvements

[Improvement direction](./Asset/Best%20Practice/Improvement.md)

## My Projects

### Foundation of Machine Learning (naive NLP, Network)

*Machine Learning Real World Data, University of Cambridge IA*

[MLRD-Cambridge_IA](https://github.com/PeterHUistyping/Machine_Learning-Real_World_Data)

- Text Classification;
- Naive Bayes
- Cross-Validation, NLP
- HMM
- Social Network

### Theoretical Machine Learning

*Theoretical Machine Learning with Problems Sets, Stanford CS229*

[ML-Stanford_CS229](https://github.com/PeterHUistyping/Stanford_CS229.Machine_Learning)

- Basic Concepts
  - Linear classifiers (Logistic Regression, GDA)
  - Stochastic Gradient Descent
  - L1 L2 Regularization
  - SVM

### Computer Vision

*Theoretical Computer Vision with Problems Sets, Stanford CS231n*

[DL-for-CV-Stanford_CS231n](https://github.com/PeterHUistyping/Stanford_CS231n-Deep_Learning-for-Computer_Vision/)

- Image Classification  + Localization $(x,y,w,h)$
  [ *Supervised Learning, Discrete label* + Regression ]
  - kNN
  - Softmax
  - classifier SVM classifier
  - CNN
  - Cross Validation
- Object Detection
- Semantic / Instance Segmentation
- Image Captioning
  - RNN, Attention, Transformer
  - Positional Encoding
- Video understanding
- Generative model (GAN, VAE)
- Self-Supervised Learning

[See more: Visual Computing](https://github.com/PeterHUistyping/Visual_Computing)

<!-- ### Case Exploration: Titanic Survival Prediction

[Titanic-ML-DataScience-Exploration](https://github.com/PeterHUistyping/Titanic-ML-DataScience-Exploration/)

- *Kaggle:Titanic Survival Prediction Exploration*
  - Updating ... -->

### More

- AI
  - [Course link](https://www.cl.cam.ac.uk/teaching/2324/ArtInt/)
  - University of Cambridge, IB
- Machine Learning and Bayesian Inference
  - [Course link](https://www.cl.cam.ac.uk/teaching/2324/MLBayInfer/)
  - University of Cambridge, II

## Reference

### OpenAI cookbook

[📝OpenAI cookbook](https://platform.openai.com/docs/introduction)

### Generative Pre-trained Transformer (GPT) from Scratch ([Andrej Karpathy](https://github.com/karpathy/))

- [▶Youtube](https://www.youtube.com/watch?v=kCc8FmEb1nY)
  - [👨‍💻Code Github repo](https://github.com/karpathy/ng-video-lecture)
- [👨‍💻nanoGPT repo](https://github.com/karpathy/nanoGPT)

**Paper**

- [📄Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [📄OpenAI GPT-3](https://arxiv.org/abs/2005.14165)
- [📝OpenAI ChatGPT blog post](https://openai.com/blog/chatgpt/)

## Library Used

Numpy, matplotlib, pandas, TensorFlow

Caffe, Keras

XGBoost, gensim
