# 1. Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score


# 2. Load the Complete 20 Newsgroups Dataset

train_data = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

test_data = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes")
)

# Display Dataset Information

print("Training Documents:", len(train_data.data))
print("Testing Documents:", len(test_data.data))
print("Number of Categories:", len(train_data.target_names))
print("All Categories:")
print(train_data.target_names)

# Display one sample document
print("\nSample Document:")
print(train_data.data[0])

# Display its category
print("\nSample Category:")
print(train_data.target_names[train_data.target[0]])


# 3. Select Four Categories

categories = [
    "sci.med",
    "rec.motorcycles",
    "comp.graphics",
    "talk.politics.misc"
]

train_data = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes"),
    categories=categories
)

test_data = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes"),
    categories=categories
)


# 4. Separate X and Y

x_train_text = train_data.data
y_train = train_data.target

x_test_text = test_data.data
y_test = test_data.target

print("\nSelected Categories:")
print(train_data.target_names)

print("Training Documents:", len(x_train_text))
print("Testing Documents:", len(x_test_text))


# 5. Model 1: Max Features = 1000

vect1 = CountVectorizer(
    stop_words="english",
    min_df=2,
    max_features=1000
)

x_train1 = vect1.fit_transform(x_train_text)
x_test1 = vect1.transform(x_test_text)

mnb1 = MultinomialNB()
mnb1.fit(x_train1, y_train)

y_pred1 = mnb1.predict(x_test1)

acc1 = accuracy_score(y_test, y_pred1)
f1_1 = f1_score(y_test, y_pred1, average="weighted")

print("\nModel 1: Max Features = 1000")
print("Actual Number of Features:", x_train1.shape[1])
print("Accuracy:", acc1)
print("F1 Score:", f1_1)


# 6. Model 2: Max Features = 3000

vect2 = CountVectorizer(
    stop_words="english",
    min_df=2,
    max_features=3000
)

x_train2 = vect2.fit_transform(x_train_text)
x_test2 = vect2.transform(x_test_text)

mnb2 = MultinomialNB()
mnb2.fit(x_train2, y_train)

y_pred2 = mnb2.predict(x_test2)

acc2 = accuracy_score(y_test, y_pred2)
f1_2 = f1_score(y_test, y_pred2, average="weighted")

print("\nModel 2: Max Features = 3000")
print("Actual Number of Features:", x_train2.shape[1])
print("Accuracy:", acc2)
print("F1 Score:", f1_2)


# 7. Model 3: Max Features = 5000

vect3 = CountVectorizer(
    stop_words="english",
    min_df=2,
    max_features=5000
)

x_train3 = vect3.fit_transform(x_train_text)
x_test3 = vect3.transform(x_test_text)

mnb3 = MultinomialNB()
mnb3.fit(x_train3, y_train)

y_pred3 = mnb3.predict(x_test3)

acc3 = accuracy_score(y_test, y_pred3)
f1_3 = f1_score(y_test, y_pred3, average="weighted")

print("\nModel 3: Max Features = 5000")
print("Actual Number of Features:", x_train3.shape[1])
print("Accuracy:", acc3)
print("F1 Score:", f1_3)


# 8. Comparison Table

comparison = pd.DataFrame({
    "Max Features": [1000, 3000, 5000],
    "Accuracy": [acc1, acc2, acc3],
    "F1 Score": [f1_1, f1_2, f1_3]
})

print("\nComparison of Models")
print(comparison.round(4))


# 9. Graph

models = ["1000", "3000", "5000"]

accuracy = [acc1, acc2, acc3]
f1_scores = [f1_1, f1_2, f1_3]

plt.figure(figsize=(8, 5))

plt.plot(models, accuracy, marker="o", label="Accuracy")
plt.plot(models, f1_scores, marker="s", label="F1 Score")

plt.title("Max Features vs Accuracy and F1 Score")
plt.xlabel("Maximum Features")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()
