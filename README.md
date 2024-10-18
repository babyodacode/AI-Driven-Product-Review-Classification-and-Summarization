# Review Sentiment and Summarization Project

## Overview

This project aims to classify customer reviews into sentiment categories (Negative, Neutral, Positive) and generate concise summaries for clusters of reviews using natural language processing techniques. The implementation leverages the T5 model for summarization and a BERT model for sentiment analysis, with K-Means clustering to group similar reviews.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models Used](#models-used)
- [Clustering](#clustering)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, install the necessary libraries using the following command:

```bash
!pip install transformers datasets scikit-learn gradio imbalanced-learn rouge-score torch nltk
```

Make sure you have the necessary libraries for data processing, machine learning, and deep learning installed.

## Usage

1. **Clone the repository** (if applicable).
2. **Load your dataset**:
   - Replace the dataset path in the code with your own dataset location.
3. **Run the script** in Google Colab or your preferred Python environment.

The Gradio interface will allow you to input customer reviews and get the predicted sentiment and summary.

## Dataset

The dataset used in this project consists of customer reviews, specifically in a CSV format. The dataset should have the following columns:

- `reviews.text`: The actual text of the review.
- `reviews.rating`: The numerical rating associated with the review (1 to 5).

The dataset should be pre-loaded into the specified path in the code.

## Preprocessing

The preprocessing steps include:

1. **Tokenization**: Splitting the text into individual words.
2. **Lowercasing**: Converting all text to lowercase for uniformity.
3. **Stopword Removal**: Removing common words (e.g., "the", "and") that do not add value.
4. **Lemmatization**: Reducing words to their base form (e.g., "running" to "run").

These steps help in reducing noise in the data, making it more suitable for analysis and model training.

## Models Used

- **BERT**: A pre-trained model used for sentiment classification. It predicts the sentiment of reviews based on the trained model.
- **T5**: A transformer model used for text summarization, generating concise summaries from the clusters of reviews.

## Clustering

K-Means clustering is employed to group similar reviews into 5 clusters. This helps in summarizing reviews based on shared sentiments or topics. 

- **Clustering Method**: K-Means with 5 clusters.
- **Clustering Performance Evaluation**: The quality of clustering is evaluated using the Silhouette Score and Davies-Bouldin Index.

## Evaluation Metrics

The following metrics are used to evaluate the sentiment classification model:

- **Accuracy**: Proportion of true results among the total number of cases examined.
- **Precision**: Measure of the accuracy of the positive predictions.
- **Recall**: Measure of the ability of the classifier to find all positive instances.
- **F1-Score**: Harmonic mean of precision and recall.

## Results

- **Confusion Matrix**: Provides a detailed breakdown of prediction results, showing how many predictions were correctly classified versus misclassified.
- **Silhouette Score**: Indicates the quality of clustering, with higher scores representing better-defined clusters.
- **Davies-Bouldin Index**: Lower values indicate better clustering quality.

Visualizations of these metrics are also generated to facilitate understanding.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you would like to contribute to this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
