# Text Summarization and Chatbot - BBC News Summary

## Table of Contents

1. Problem Statement
2. Initialization: Loading Data
3. EDA (Exploratory Data Analysis)
   - Distribution of Number of Articles in Each Category
4. Preprocessing
   - Word Cloud
   - Topic Modeling
   - N-grams Analysis
5. Transformation
6. Clustering
   - Silhouette with K-means
   - Plot the Clusters for Bag of Words representation with K-means
   - Plot the Clusters for TF-IDF representation with K-means
   - Plot the Clusters for Word2Vec representation with K-means
7. Hierarchical Clustering
   - Hierarchical Clustering with BOW
   - Hierarchical Clustering with TF-IDF
   - Hierarchical Clustering with Word Embedding
8. Silhouette with Hierarchical Clustering
9. Save Model with the Highest Accuracy - K-Means with Word2Vec
10. Load Saved Model
11. Extractive Text Summarization
12. Evaluation of Extractive Summarization
13. Classification
    - ML Models based on BOW
14. Abstractive Summarization
15. Evaluation of Abstractive Summarization

## 1. Problem Statement
In this project, we aim to perform text summarization using BBC News Summary data. The goal is to generate accurate and concise summaries of news articles using both extractive and abstractive summarization techniques. Additionally, we plan to build a chatbot that interacts with users, accepts queries, and provides relevant summaries for the requested topics or news articles.

## 2. Initialization: Loading Data
We start by loading the BBC News Summary dataset, which contains articles from various categories, such as business, entertainment, politics, sports, and technology.
Uploud the data into your drive
dataset link: https://www.kaggle.com/datasets/pariza/bbc-news-summary
## 3. EDA (Exploratory Data Analysis)
We explore the dataset to gain insights into the distribution of articles in each category.

## 4. Preprocessing
Before performing summarization, we preprocess the text data, including cleaning, tokenization, and removing stopwords and special characters. We also visualize the data using a word cloud and perform topic modeling and N-grams analysis to understand the key topics and frequently occurring phrases.

## 5. Transformation
We convert the preprocessed text data into various representations, including Bag of Words (BOW), Term Frequency-Inverse Document Frequency (TF-IDF), and Word Embedding (Word2Vec).

## 6. Clustering
To analyze the similarity between articles, we apply clustering algorithms such as K-means and Hierarchical Clustering to group similar articles together. We use the Silhouette score to evaluate the clustering performance.

## 7. Hierarchical Clustering
We further explore Hierarchical Clustering with different representations to gain deeper insights into the article similarities.

## 8. Silhouette with Hierarchical Clustering
We evaluate the Hierarchical Clustering performance using the Silhouette score.

## 9. Save Model with the Highest Accuracy - K-Means with Word2Vec
We save the K-means model with the highest accuracy obtained using Word2Vec representation.

## 10. Load Saved Model
We load the previously saved K-means model for future use.

## 11. Extractive Text Summarization
We perform extractive text summarization, where we select sentences from the original article that best represent its content. We evaluate the quality of the generated summaries.

## 12. Evaluation of Extractive Summarization
We use evaluation metrics like ROUGE and BLEU scores to assess the extractive summarization performance.

## 13. Classification
We build machine learning models based on BOW representation to classify articles into different categories.

## 14. Abstractive Summarization
We implement abstractive summarization techniques, such as transformer-based models, to generate more human-like summaries by rephrasing and paraphrasing.

## 15. Evaluation of Abstractive Summarization
We evaluate the abstractive summarization performance using evaluation metrics and user feedback.

Throughout the project, we aim to achieve accurate and informative text summarization results and develop an interactive chatbot that can assist users in obtaining concise and relevant summaries for their news queries. The use of multiple representations and clustering techniques ensures that we explore various dimensions of summarization and strive for a comprehensive and high-quality solution.# Text Summarization and Chatbot - BBC News Summary

## License

[BBC Summary](https://www.kaggle.com/datasets/pariza/bbc-news-summary)
