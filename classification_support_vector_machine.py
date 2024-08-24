import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Preprocess text data
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Convert to lowercase - standardization
    standardization = text.lower()

    # Tokenize words - tokenization
    tokenization = word_tokenize(standardization)

    # Remove stopwords and apply lemmatization - stopword removal and lemmatization
    processed_words = [lemmatizer.lemmatize(word) for word in tokenization if word.isalnum() and word not in stop_words]
    
    stopwords_lemmatization = ' '.join(processed_words)
    
    return standardization, tokenization, stopwords_lemmatization

# Main function for training and testing - classification
def train_test_sentiment_analysis(file_path, text_column, label_column, test_size=0.3):
    # Load data
    df = pd.read_excel(file_path, sheet_name='comment')

    # Preprocess text
    df[['standardization', 'tokenization', 'stopwords_lemmatization']] = df[text_column].apply(preprocess_text).apply(pd.Series)
    
    # Display the table with original and processed texts
    print("Text Comparison Table:")
    print(df[['comment_translated', 'standardization', 'tokenization', 'stopwords_lemmatization']].head())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['stopwords_lemmatization'], df[label_column], test_size=test_size, random_state=42
    )

    # Create a pipeline with TfidfVectorizer and Support Vector Machine (SVM) classifier
    model = make_pipeline(TfidfVectorizer(), SVC(kernel='poly', degree=3))  # Using Polynomial kernel

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    # Calculate the counts for each category
    tn, fp, fn, tp = conf_matrix.ravel()

    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Display metrics
    metrics_table = pd.DataFrame({
        'Predicted Instance': ['Positive', 'Negative', 'Metrics'],
        'Actual Positive': [f'True Positive (TP): {tp}', f'False Negative (FN): {fn}', f'Recall: {recall:.2f}'],
        'Actual Negative': [f'False Positive (FP): {fp}', f'True Negative (TN): {tn}', f'Specificity: {specificity:.2f}'],
        'Metrics': [f'Precision: {precision:.2f}', f'Negative Predicted Value (NPV): {npv:.2f}', f'Accuracy: {accuracy:.2f}, F1-score: {f1_score:.2f}']
    })
    print(metrics_table)

    # Plot confusion matrix with counts and accuracy
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Add counts and accuracy to the plot
    plt.text(0.5, -0.1, f'True Negative (TN): {tn}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.5, -0.15, f'False Positive (FP): {fp}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.5, -0.2, f'False Negative (FN): {fn}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.5, -0.25, f'True Positive (TP): {tp}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.5, -0.3, f'Accuracy: {accuracy:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)

    plt.show()

file_path = 'ugc_data_.xlsx'
text_column = 'comment_translated'
label_column = 'result'
train_test_sentiment_analysis(file_path, text_column, label_column)
