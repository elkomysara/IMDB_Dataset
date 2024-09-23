
# Sentiment Analysis on IMDB Movie Reviews

## Overview

This project focuses on **Sentiment Analysis** using the **IMDB Dataset** of 50,000 movie reviews. The goal of this project is to build a machine learning model that can classify movie reviews as either **positive** or **negative** based on the text of the review.

## Dataset

The dataset is sourced from Kaggle: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). It contains:
- **50,000 reviews** labeled as either positive or negative.
- Each review consists of free-form text data.

### Features:
- **review**: The text of the movie review.
- **sentiment**: The sentiment label (`positive` or `negative`).


## Project Workflow

### 1. Data Loading and Exploration
- The dataset is loaded into a Pandas DataFrame.
- Basic exploration is done to understand the distribution of sentiments and check for missing values.

### 2. Data Cleaning
- The reviews are preprocessed by:
  - Converting text to lowercase.
  - Removing non-alphabet characters.
  - Removing extra whitespace.
  - Removing HTML tags.

### 3. Text Vectorization
- Text data is transformed into numerical format using **TF-IDF (Term Frequency-Inverse Document Frequency)** to prepare it for the machine learning model.

### 4. Model Training
- A **Logistic Regression** model is trained to classify reviews into positive or negative categories.
- The dataset is split into training and testing sets, with **80%** used for training and **20%** for testing.

### 5. Model Evaluation
- The trained model is evaluated using various metrics, including:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
- A **confusion matrix** is used to visualize the performance of the model.

## Installation

To run this project, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/your-username/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Download the IMDB dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place the `IMDB Dataset.csv` file in the `data` directory.

### 4. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/Sentiment_Analysis.ipynb
```

## Model Performance

The Logistic Regression model achieves the following results on the test dataset:

- **Accuracy**: 88%
- **Precision**: 89%
- **Recall**: 87%
- **F1-Score**: 88%

## Future Improvements

- Experiment with more advanced machine learning models like **Naive Bayes**, **SVM**, or **Deep Learning** using **LSTM** or **BERT**.
- Perform **hyperparameter tuning** for improving model performance.
- Add support for **neutral sentiment** classification.

## Author
Sara Elkomy

## License

This project is licensed under the MIT License.
