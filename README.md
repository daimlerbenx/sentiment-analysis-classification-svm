# Sentiment Analysis Classification with SVM

This repository contains a Python script for performing sentiment analysis on text data using Support Vector Machines (SVM). The script preprocesses text data, trains a classifier, and evaluates its performance using metrics and visualizations.

## Dependencies

The script requires the following Python libraries:

- `pandas`: For data manipulation and analysis
- `scikit-learn`: For machine learning models and utilities
- `nltk`: For natural language processing tasks
- `seaborn`: For data visualization
- `matplotlib`: For plotting

You can install the necessary libraries using pip:

```bash
pip install pandas scikit-learn nltk seaborn matplotlib
```

## Script Overview

### 1. **Preprocess Text Data**

The `preprocess_text` function performs the following steps on the input text:
- **Standardization**: Converts text to lowercase.
- **Tokenization**: Splits the text into individual words.
- **Stopword Removal and Lemmatization**: Removes common stopwords and applies lemmatization to reduce words to their base form.

The function returns:
- Standardized text
- Tokenized words
- Processed text with stopwords removed and lemmatized

### 2. **Train and Test Sentiment Analysis**

The `train_test_sentiment_analysis` function performs the following steps:
1. **Load Data**: Reads data from an Excel file.
2. **Preprocess Text**: Applies the `preprocess_text` function to the text column and adds the processed results to the DataFrame.
3. **Split Data**: Divides the data into training and testing sets.
4. **Create and Train Model**: Uses a pipeline with `TfidfVectorizer` and an SVM classifier with a polynomial kernel.
5. **Make Predictions**: Generates predictions on the test set.
6. **Evaluate Model**: Computes accuracy, classification report, and confusion matrix. Displays additional metrics such as precision, recall, specificity, and F1-score.
7. **Plot Confusion Matrix**: Visualizes the confusion matrix with counts and accuracy.

### Usage

To run the script, ensure you have an Excel file named `ugc_data_.xlsx` with columns for text data and labels. Update the `file_path`, `text_column`, and `label_column` variables if needed.

Run the script using:

```bash
python classification_support_vector_machine.py
```

## Results

The script will output:
- A table comparing the original and processed text.
- Metrics for model performance.
- A confusion matrix plot.

Feel free to adjust any details based on your specific needs or preferences!
