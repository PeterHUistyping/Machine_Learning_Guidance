# Machine Learning Guidance

The Repo integrates a collection of my *Machine Learning and Data Science Projects*, including both Theoretical and Practical/ Applied.

## Machine Learning Knowledge Sharing

### ML Categories

| Unsupervised Learning                 | Supervised Learning                            |
| ------------------------------------- | ---------------------------------------------- |
| **Continuous**                  |                                                |
| Clustering & Dimensionality Reduction | Regression                                     |
| ○ SVD                                | ○ Linear / Polynomial                         |
| ○ PCA                                | ○ Non-Linear Regression                       |
| ○ K-means                            | Decision Trees                                 |
|                                       | Random Forest                                  |
| **Discrete / Categorical**      |                                                |
| Association Analysis                  | Classification                                 |
| ○ Apriori                            | Generative Model (Joint Prob.)                 |
| ○ FP-Growth                          | ○ Naive Bayes                                 |
| Hidden Markov Model                   | Discriminative Model (Conditional Prob.)      |
|                                       | ○ Logistic Regression   ○ Perceptron  ○ SVM |
|                                       | ○ KNN / Trees                                 |

**Semi-Supervised Learning** 

- uses a small portion of labeled data and lots of unlabeled data to train a predictive model
- iteratively generate pseudo-labels for a new dataset

**Reinforcement Learning**

- Markov Decision Process
- Q learning
- Value-based           V(s)

  - the agent is expecting a long-term return of the current states under policy π
- Policy-based

  - the action performed in every state helps you to gain maximum reward in the future
  - Deterministic: For any state, the same action is produced by the policy π
  - Stochastic: Every action has a certain probability
- Model-based

  - create a virtual model for each environment
  - the agent learns to perform in that specific environment

### Feature Engineering
- Feature Selection
- Imputation
- Handling Outliers
  - Removal, Replacing values, Capping, Discretization 
- Encoding
  - Integer Encoding
  - One-Hot Encoding (enum -> binary)
- Scaling 
  - Normalization, min-max/ 0-1
  - Standardization
## Links to relevant Project / Repo [submodule]

#### My Projects

### Foundation of Machine Learning (naive NLP, Network)

[CambridgeIA-Machine_Learning-Real_World_Data_with_Python_src](https://github.com/PeterHUistyping/Machine_Learning-Real_World_Data)

- *Machine Learning Real World Data, University of Cambridge IA*
  - Text Classification using ML Naive Bayes;
  - Cross-Validation, NLP;
  - HMM;
  - Social Network

### Theoretical Machine Learning

[Stanford_CS229.Machine_Learning](https://github.com/PeterHUistyping/Stanford_CS229.Machine_Learning)

- *Theoretical Machine Learning with Problems Sets, Stanford CS229*
  - Linear classifiers (Logistic Regression, GDA)
  - Stochastic Gradient Descent
  - L1 L2 Regularization
  - SVM

### Computer Vision

[Stanford_CS231n-Deep_Learning-for-Computer_Vision](https://github.com/PeterHUistyping/Stanford_CS231n-Deep_Learning-for-Computer_Vision/)

- *Theoretical Computer Vision with Problems Sets, Stanford CS231n*
  - kNN
  - Softmax
  - SVM classifier
  - Cross Validation

### Case Exploration: Titanic Survival Prediction

[Titanic-ML-DataScience-Exploration](https://github.com/PeterHUistyping/Titanic-ML-DataScience-Exploration/)

- *Kaggle:Titanic Survival Prediction Exploration*
  - Updating

#### Others' Projects


## Library Used

Numpy, matplotlib, pandas, TensorFlow

Caffe, Keras

XGBoost, gensim