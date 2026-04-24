# NLP Project Group 4: Fake News Detection

## Overview
This project explores fake news classification with Natural Language Processing (NLP). We compare classical sparse-text methods such as `BoW` and `TF-IDF` with embedding-based approaches such as `Word2Vec` and `BERT`, and evaluate how different model families behave across these feature types.

The project is built around one main question:

- Which combination of text representation and model gives the best performance?

## Objective
- Classify news headlines as `Real` or `Fake`
- Compare multiple preprocessing, feature, and model combinations
- Identify the best-performing and most reliable pipeline
- Understand the tradeoff between interpretability, efficiency, and predictive performance

## Hypothesis
- Embedding-based methods, especially `BERT`, can outperform classical sparse-text approaches when paired with suitable models such as Neural Networks
- Classical sparse-text pipelines, especially `TF-IDF + Logistic Regression`, remain strong and interpretable baselines

## Dataset

### Training data
- File: training_data_lowercase.csv
- Contains:
  - headline text
  - class label

### Test data
- File: testing_data_lowercase_nolabels.csv
- Contains:
  - headline text only
- Used for final prediction

## Experimental Design

### Feature families

#### Sparse-text representations
- `BoW`
- `TF-IDF`
- `TF-IDF` with log-scaled term frequency

#### Embedding representations
- `Word2Vec`
- `BERT`

### Preprocessing variants for sparse-text pipelines
- `split_lemma`
- `split_stem`
- `token_lemma`
- `token_stem`

### Model families
- `Naive Bayes`
- `Logistic Regression`
- `Random Forest`
- `SVM`
- `XGBoost`
- `Neural Network`

### Total configurations
- `72` sparse-text pipelines: `4 × 3 × 6`
- `12` embedding pipelines: `2 × 6`
- Total: `84` experiment configurations

## Workflow
The project follows this general pipeline:

`Preprocessing -> Feature Extraction -> Model Training -> Evaluation`

To keep the experiments consistent, we used:

- `Pipeline`
- `RandomizedSearchCV`

This helped us:

- reduce data leakage risk
- apply the same tuning logic across models
- compare pipelines more fairly

## What Was Tuned

### Sparse-text pipelines
For the sparse-text family, tuning was applied to both the vectorizer and the classifier.

#### Vectorizer parameters
- `max_features`
- `min_df`
- `max_df`

#### Classifier parameters
- `Naive Bayes`: `alpha`
- `Logistic Regression`: `C`
- `Random Forest`: `n_estimators`, `max_depth`
- `SVM`: `C`
- `XGBoost`: `n_estimators`, `max_depth`, `learning_rate`
- `Neural Network`: `hidden_layer_sizes`, `alpha`

### Embedding pipelines
For `Word2Vec` and `BERT`, the embedding representations were kept fixed and tuning was applied only to the classifier on top of the embeddings.

#### Fixed embedding setup
- `Word2Vec`: used as a fixed embedding generator
- `BERT`: used as a fixed pretrained model (`distilbert-base-uncased`)
- The embedding generators themselves were not fine-tuned in the final comparison

#### Classifier parameters tuned on embeddings
- `Gaussian Naive Bayes`: `var_smoothing`
- `Logistic Regression`: `C`
- `Random Forest`: `n_estimators`, `max_depth`
- `SVM`: `C`
- `XGBoost`: `n_estimators`, `max_depth`, `learning_rate`
- `Neural Network`: `hidden_layer_sizes`, `alpha`

## Evaluation
Each experiment was assessed using:

- training accuracy
- validation accuracy
- cross-validation accuracy

We also used the results to examine:

- overfitting through the train-validation gap
- model stability across setups
- the relative importance of preprocessing, feature choice, and model choice

## Main Findings
- Feature representation mattered more than small preprocessing changes
- `TF-IDF` consistently outperformed or matched `BoW`
- Embedding-based methods, especially `BERT`, achieved the best overall results
- `Logistic Regression` and `SVM` remained strong and stable classical baselines
- `Neural Network` benefited the most from embedding features
- `Random Forest` and `XGBoost` were more prone to overfitting in several setups

## Best Overall Result
- Feature family: `embedding`
- Variant: `bert_standard`
- Model: `Neural Network`

This setup achieved the strongest validation performance in the full comparison.

## Strong Baseline
- `TF-IDF + Logistic Regression`

This remained one of the most useful baselines because it was:

- strong
- stable
- efficient
- interpretable

## Project Files

### Main notebooks
- [NLP_Project_Group4.ipynb]


### Presentation material
- `NLP_G4_Slides.pptx`


### Output: Test prediction (saved as test_predictions_based_on_BERT_Neural_Network.csv) based on the best performed setup:
Best overall setup:
feature_family           embedding
variant              bert_standard
feature                       bert
model               Neural Network
train_accuracy                 1.0
val_accuracy               0.97526
best_cv_accuracy          0.969071
Name: 0, dtype: object

Best parameters:
{'model__hidden_layer_sizes': (128,), 'model__alpha': 0.001}

Validation classification report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3515
           1       1.00      1.00      1.00      3316

    accuracy                           1.00      6831
   macro avg       1.00      1.00      1.00      6831
weighted avg       1.00      1.00      1.00      6831


First 10 test predictions:
   predicted_label
0                0
1                0
2                1
3                0
4                1
5                0
6                1
7                0
8                1
9                1


## Figures
The project folder also includes exported figures used for analysis and presentation, for example:

- `ModelComparison.png`
- `Most_Influential_words.png`
- `Overfitting_plots.png`
- `Training_Validation_byVecto.png`
- `Training_validation_bypreprossing.png`
- `Figures/`


## Tech Stack
- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- NLTK
- gensim
- transformers
- PyTorch
- matplotlib
- seaborn
