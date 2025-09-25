# 🎬 Movie Sentiment Analysis

## Overview

This project performs sentiment analysis on a large dataset of movie reviews using machine learning and natural language processing techniques. The model classifies reviews as **positive** or **negative** based on their text content.

## Features

- Reads and analyzes a clean CSV dataset (`movie.csv`) of movie reviews.
- Exploratory Data Analysis (EDA) with visualizations:
  - Distribution of sentiment labels
  - Review length histogram
  - Word count histogram
- Text preprocessing and feature extraction using **TF-IDF Vectorization**.
- Machine learning model training using **Multinomial Naive Bayes**.
- Model evaluation with accuracy and classification report.
- Predicts sentiment for new, unseen reviews.

## Performance Metrics

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.84      0.89      0.87      5994
           1       0.88      0.83      0.86      6006

    accuracy                           0.86     12000
   macro avg       0.86      0.86      0.86     12000
weighted avg       0.86      0.86      0.86     12000
```

- **Accuracy:** 0.86
- **Precision (Negative):** 0.84
- **Recall (Negative):** 0.89
- **F1-score (Negative):** 0.87
- **Precision (Positive):** 0.88
- **Recall (Positive):** 0.83
- **F1-score (Positive):** 0.86

## Usage

1. Place your `movie.csv` file in the same directory as `moviesentimentanalysis.py`.
2. Run the script:
   ```bash
   python moviesentimentanalysis.py
   ```
3. The script will:
   - Print the dataset and column info
   - Show EDA plots (label distribution, review length, word count)
   - Train and evaluate the sentiment analysis model
   - Print accuracy and classification report
   - Predict sentiment for sample new reviews

## Example Output

```
==================================================
🎬 Sentiment Analysis of Movie Reviews
==================================================

Dataset:
   ... (prints first few rows of movie.csv) ...

Sentiment analysis model trained!

Accuracy: 0.86

Classification Report:
              precision    recall  f1-score   support
           0       0.84      0.89      0.87      5994
           1       0.88      0.83      0.86      6006
    accuracy                           0.86     12000
   macro avg       0.86      0.86      0.86     12000
weighted avg       0.86      0.86      0.86     12000

--- Testing with new reviews ---
Review: "The story was incredible and the visuals were stunning." -> Sentiment: Positive
Review: "It was a very dull and slow-paced film." -> Sentiment: Negative
Review: "A decent movie, not great but not bad either." -> Sentiment: Negative
```

## Dependencies

- numpy
- pandas
- scikit-learn
- seaborn
- matplotlib

Install with:
```bash
pip install numpy pandas scikit-learn seaborn matplotlib
```

## Acknowledgments

Special thanks to **Srihari Sir (IIT Guwahati Faculty)** for providing the foundational code, educational guidance, and opportunities to learn advanced machine learning concepts.  
Thanks also to **Masai School** for supporting this learning journey.

---

*Most of the code and structure was created by Srihari Sir, with educational spaces and comments for teaching. My contribution was primarily in learning, understanding, and completing the exercises provided.*

---
