# Bank Reputation Sentiment Analysis with BERT, LSTM, & SVM
This project compares the performance of traditional Machine Learning (SVM) and Deep Learning models (LSTM, BERT) in analyzing public sentiment towards banking institutions. The study utilizes the "Customers Reviews on Banks" dataset from Hugging Face to classify reviews into 1-5 star ratings, specifically addressing challenges with imbalanced data in the financial sector.

## 1. Data Collection & Preprocessing 🧹
The initial phase involves gathering and cleaning textual data to prepare it for Natural Language Processing (NLP) tasks.

Data Source: The dataset is obtained from Hugging Face ("Customers Reviews on Banks"), containing 19.3k user-generated reviews spanning 48 distinct banking institutions from 2017 to 2023.

EDA: Exploratory Data Analysis revealed a severe class imbalance, with the majority of reviews being 1-star (16,476 reviews) and very few in the 3-star (292) or 4-star (373) categories.

Splitting: The dataset is divided into an 80:20 ratio for training and testing to ensure unbiased evaluation.

Preprocessing Pipeline: The raw text undergoes a rigorous cleaning process:

Text Cleaning: Removal of punctuation, special characters, and excessive whitespace.

Tokenization: Segmenting reviews into individual word units.

Stopword Removal: Eliminating common, non-informative words.

Lemmatization: Converting words to their base forms to reduce vocabulary size.

## 2. Feature Engineering & Architectures ⚙️
Three distinct modeling approaches were implemented to compare performance across different levels of complexity.

Feature Extraction:

TF-IDF: Used for SVM and LSTM models to capture the relative importance of terms across the corpus.

Contextual Embeddings: Used for BERT to capture semantic nuances and long-range dependencies.

Model Architectures:

SVM (Baseline): A traditional Support Vector Machine used for its effectiveness in high-dimensional text data.

LSTM: A Long Short-Term Memory network designed to process sequential data and learn context from tokenized sequences.

BERT: A Transformer-based architecture (Pretrained) fine-tuned on the dataset to leverage deep contextual understanding.

## 3. Training & Handling Imbalance 🧑‍💻
Given the highly imbalanced nature of the dataset, specific strategies were employed during the training phase.

Class Weighting: All three models incorporate class weighting during training. This assigns a higher penalty to misclassified minority classes (like 3 and 4 stars) to prevent the models from becoming biased toward the majority 1-star class.

Hyperparameter Tuning: Parameter tuning was performed on both unweighted and weighted versions of the models to optimize performance.

Evaluation Metrics: Instead of relying solely on Accuracy (which can be misleading), the project focuses on Precision, Recall, and F1-Score to truly measure the model's ability to detect minority sentiments.

## 4. Results & Analysis 📊
The project highlights the trade-offs between model complexity and performance on imbalanced financial data.

The "Accuracy Trap": Traditional models (SVM and LSTM) achieved high accuracy (~90%) but failed significantly on minority classes, often predicting the majority class exclusively. For example, unweighted SVM had a Precision/Recall of 0.00 for minority classes.

BERT Performance: BERT proved to be the superior architecture. It outperformed TF-IDF based methods, achieving the highest Macro F1-score of 0.46 and demonstrating a stronger ability to distinguish extreme sentiments (1 and 5 stars).

Challenge with Intermediate Ratings: All models, including hypertuned BERT, struggled to classify intermediate ratings (2, 3, and 4 stars), suggesting these classes lack distinct linguistic features compared to the extremes.
