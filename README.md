# Machine Learning Guidance

A collection of my *Machine Learning and Data Science Projects*, 
- including both Theoretical and Practical (Applied) ML,
- references (paper, ebook, repo, tool, etc), ranging from beginner to advanced.
<!-- that's interesting and helpful attached. -->

## Generative vs Discriminative Model

Given the training data set $D = \{ ( x_i ; y_i ) | i ≤ N ∈ Z \}$, where $y_i$ is the corresponding output for the input $x_i$.

| Aspect\Model | Generative                                             | Discriminative                                 |
| ------------ | ------------------------------------------------------ | ---------------------------------------------- |
| Learn obj    | $P(x,y)$      <br /> Joint probability               | $P(y\vert x)$ <br /> Conditional probability |
| Formulation  | class prior/conditional<br />$P(y)$, $P(x\vert y)$ | likelihood<br />$P(y\vert x)$               |
| Result       | not direct (Bayes)<br /> $P(y\vert x)$               | direct classification                          |
| Examples     | Naive Bayes, HMM                                       | Logistic Reg, SVM,<br />DNN                    |

Reference: [Generative and Discriminative Model](http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf), Professor Andrew NG

## Types of Learning

- Learner $L(X_I)=h \in \mathcal{H}$
  - Input training data $X_I$, where $x_i \in \R$
  - Hypothesis $h_{\omega}: X \in R^n \rightarrow Y$, with weights $\omega$.
    - mapping attributes vectors $X$ to labels/output $Y=\{y_1,...,y_n\}$
    - For NN, $h(x)=f(\omega;x)$, explicitly parameterized by $\omega$
    - For Generative model $f: Z \rightarrow X$, $Z$ is the latent variable

| Output \ Type                                  | Unsupervised                                        | Supervised                                |
| ---------------------------------------------- | --------------------------------------------------- | ----------------------------------------- |
| *Continuous*  $Y$ <br /> $=\R$           | ***Clustering & <br />Dim Reduction***      | ***Regression***                  |
|                                                | ○ SVD                                              | ○ Linear / Polynomial                    |
|                                                | ○ PCA                                              | ○ Non-Linear Regression                  |
|                                                | ○ K-means                                          | ○ Decision Trees                        |
|                                                | ○ GAN<br />○ VAE ○ Diffusion                   | ○ Random Forest                         |
| *Discrete*  $Y$ <br /> $=\{Categories\}$ | ***Association /<br /> Feature Analysis*** | ***Classification***              |
|                                                | ○ Apriori                                          | ○ Bayesian       ○ SVM              |
|                                                | ○ FP-Growth                                        | ○ Logistic Regression<br />○ Perceptron |
|                                                | ○ HMM                                              | ○ kNN / Trees                            |

And more,

| Aspect \ Type | Semi-Supervised  | Reinforcement               |
| ------------- | ---------------- | --------------------------- |
| Learn from    | Labels available | Rewards                     |
| Methods       | pseudo-labels    | ○ Q learning              |
|               | iteratively      | ○ Markov Decision Process |

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

## Geometric Deep Learning

The bigger picture of learning with invariances and symmetries:

| Domain              | Structure                 | Symmetry / Bias         | Example                          |
| ------------------- | ------------------------- | ----------------------- | -------------------------------- |
| Images              | 2D grid                   | Translation equivariant | CNNs                             |
| Sequences           | 1D sequence               | Order-aware             | RNNs, Transformers               |
| Sets / Point Clouds | Unordered set             | Permutation invariant   | Deep Sets, PointNet              |
| Graphs              | Nodes + edges             | Permutation equivariant | GNNs, Graph Isomorphism Networks |
| Manifolds / Spheres | 2D surface embedded in 3D | Rotation equivariant    | Spherical CNNs                   |

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

## Inference

| Aspect                                         | Bayesianism                                                                                   | Frequentism                                                           |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Interpretation of Probability                 | A measure of belief or uncertainty                                                            | The limit of relative frequencies<br />in repeated experiments        |
| Methods                                        | Prior knowledge and updates beliefs (Bayes')<br />to obtain posterior distributions           | Hypothesis testing, MLE, confidence intervals                        |
| Treatment of Uncertainty<br />Random Variables | Parameters                                                                                    | Data set                                                              |
| Handling of Data                               | useful when prior information is available or<br />when the focus is on prediction intervals. | often requires larger sample sizes                                    |
| Flexibility                                    | flexible model,<br />allow updating models for new data                                      | more rigid, on specific statistical methods                           |
| Computational Complexity                       | can be intensive computation,<br />for models with high-dim parameter spaces                 | simpler computation and<br />may be more straightforward in practice |

#### Empiricism

Applied ML Best Practice

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

<!-- ### Foundation of Machine Learning (naive NLP, Network) -->

[Machine Learning Real World Data, University of Cambridge IA](https://github.com/PeterHUistyping/Machine_Learning-Real_World_Data)

- Text Classification; Naive Bayes; Cross-Validation, 
- HMM; Social Network

<!-- ### Theoretical Machine Learning -->

[Theoretical Machine Learning with Problems Sets, Stanford CS229](https://github.com/PeterHUistyping/Stanford_CS229.Machine_Learning)

- Linear classifiers (Logistic Regression, GDA), SVM, etc
- Stochastic Gradient Descent; L1 L2 Regularization

<!-- ### Computer Vision -->

[Deep Learning for Computer Vision with Problems Sets, Stanford CS231n](https://github.com/PeterHUistyping/Stanford_CS231n-Deep_Learning-for-Computer_Vision/)

- Image Classification  + Localization $(x,y,w,h)$
  [ *Supervised Learning, Discrete label* + Regression ]
  - kNN; Softmax; classifier SVM classifier; CNN
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

  
- [Data Science](https://www.cl.cam.ac.uk/teaching/2324/DataSci/) | Uni. of Cambridge, Undergraduate course.
- [AI](https://www.cl.cam.ac.uk/teaching/2324/ArtInt/) | Uni of Cambridge, IB
  - Search, Game, CSPs, Knowledge representation and Reasoning, Planning, NN.
- [Machine Learning and Bayesian Inference](https://www.cl.cam.ac.uk/teaching/2324/MLBayInfer/) | Uni of Cambridge, Undergraduate course.
  - Linear classifiers (SVM), Unsupervised learning (K-means,EM), Bayesian networks
- [Geometric Deep Learning](https://geometricdeeplearning.com/) | Cambridge, Oxford Master's courses.

## Reference

[📝OpenAI cookbook](https://platform.openai.com/docs/introduction)

Generative Pre-trained Transformer (GPT) from Scratch ([Andrej Karpathy](https://github.com/karpathy/))
- [▶Youtube](https://www.youtube.com/watch?v=kCc8FmEb1nY)
  - [👨‍💻Code Github repo](https://github.com/karpathy/ng-video-lecture)
- [👨‍💻nanoGPT repo](https://github.com/karpathy/nanoGPT)

**Paper**

- [📄Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [📄OpenAI GPT-3](https://arxiv.org/abs/2005.14165)
- [📝OpenAI ChatGPT blog post](https://openai.com/blog/chatgpt/)
- etc.

## Library

- Numpy, matplotlib, pandas, TensorFlow
- Caffe, Keras
- XGBoost, gensim
