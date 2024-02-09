```markdown
# Sentiment Analysis Model Creation

## Overview

This project demonstrates the creation of a sentiment analysis model using Python and scikit-learn library. The model is trained on a dataset of movie reviews labeled with sentiment (positive or negative). After training, the model is capable of predicting the sentiment of new movie reviews.

## Prerequisites

Make sure you have Python installed on your system. You can download and install Python from the official website: [Python Downloads](https://www.python.org/downloads/)

Additionally, you need to install the required Python libraries. You can install them using pip, which is the package installer for Python. Run the following command:

```
pip install scikit-learn nltk
```

## Project Structure

- **Model_Creation.py**: This Python script contains the code for training the sentiment analysis model. It defines the training and test data, pipelines for classifiers, model training, and prediction.

## Usage

1. Clone the repository to your local machine:

```
git clone <repository_url>
```

2. Navigate to the project directory:

```
cd <project_directory>
```

3. Run the Model_Creation.py script:

```
python Model_Creation.py
```

This will train the sentiment analysis model, make predictions on test data, and print the predicted sentiments for each review.

## Sample Output

```
Review: The movie was fantastic!
Predicted Sentiment: Positive Sentiment

Review: I hated every minute of it.
Predicted Sentiment: Negative Sentiment

Review: The performances were mediocre.
Predicted Sentiment: Negative Sentiment

Review: The plot twist caught me off guard.
Predicted Sentiment: Positive Sentiment

Review: I wouldn't recommend this movie to anyone.
Predicted Sentiment: Negative Sentiment

Review: It was better than I expected.
Predicted Sentiment: Positive Sentiment

Review: The ending left me wanting more.
Predicted Sentiment: Negative Sentiment

Review: The special effects were amazing!
Predicted Sentiment: Positive Sentiment

Review: I couldn't take my eyes off the screen.
Predicted Sentiment: Positive Sentiment

Review: The dialogue felt natural and engaging.
Predicted Sentiment: Negative Sentiment

```

## Accuracy

The accuracy of the sentiment analysis model can be calculated by comparing the predicted sentiments with the actual sentiments of the test data. The accuracy score indicates the proportion of correct predictions out of the total predictions made.

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, feel free to open an issue or create a pull request.
```
