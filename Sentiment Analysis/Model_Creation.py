from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import re

# Define training data
reviews_train = [
    "The acting in this movie was superb.",
    "I couldn't stand the lead actor's performance.",
    "The storyline kept me engaged throughout the film.",
    "The script was poorly written and predictable.",
    "This is one of the best movies I've seen in years!",
    "The cinematography was breathtaking.",
    "I found the plot to be confusing and hard to follow.",
    "The characters were well-developed and relatable.",
    "I was disappointed by the lackluster ending.",
    "The soundtrack perfectly complemented the mood of the film.",
    "The special effects were underwhelming.",
    "The pacing of the movie felt off and dragged in parts.",
    "I was blown away by the twist ending!",
    "The dialogue felt forced and unnatural.",
    "The movie was a complete waste of time.",
    "I laughed out loud multiple times during this comedy.",
    "The film left me feeling emotionally drained.",
    "The chemistry between the main characters was palpable.",
    "I was on the edge of my seat the entire time!",
    "The plot was original and kept me guessing until the end."
]

labels_train = [
    1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
    0, 0, 1, 0, 0, 1, 0, 1, 1, 1
]

# Define test data
reviews_test = [
    "The movie was fantastic!",
    "I hated every minute of it.",
    "The performances were mediocre.",
    "The plot twist caught me off guard.",
    "I wouldn't recommend this movie to anyone.",
    "It was better than I expected.",
    "The ending left me wanting more.",
    "The special effects were amazing!",
    "I couldn't take my eyes off the screen.",
    "The dialogue felt natural and engaging."
]

labels_test = [
    1, 0, 0, 1, 0, 1, 1, 0, 1, 0
]

# Define pipelines for classifiers
MNB_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))), 
                         ('clf', MultinomialNB(alpha=1.0, fit_prior=True)),
                        ])

KNN_pipeline = Pipeline([('vect', CountVectorizer()), 
                         ('clf', KNeighborsClassifier(n_neighbors=5)),
                        ])
                        
LR_pipeline = Pipeline([('vect', CountVectorizer()), 
                        ('tfidf', TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)),
                        ('clf', LogisticRegression(warm_start=True, random_state=1)),
                       ]) 
                     
# Define Voting Classifier
eclf = VotingClassifier(estimators=[('MNB', MNB_pipeline), ('KNN', KNN_pipeline), ('LR', LR_pipeline)], voting='soft', weights=[3, 2, 3])

# Fit the ensemble classifier
eclf.fit(reviews_train, labels_train)

# Predict using the ensemble classifier
pred = eclf.predict(reviews_test)

# Map labels to sentiment descriptions
sentiment_map = {
    0: "Negative Sentiment",
    1: "Positive Sentiment"
}

# Calculate accuracy
accuracy = accuracy_score(labels_test, pred)

# Print predictions and accuracy
for review, prediction in zip(reviews_test, pred):
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {sentiment_map[prediction]}")
    print()

print(f"Accuracy: {accuracy}")
