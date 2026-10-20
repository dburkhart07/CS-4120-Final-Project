import sklearn_crfsuite
from sklearn_crfsuite import metrics
import json


def word2features(sentence, i):
    word = sentence[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:3]': word[:3],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.has_hyphen': '-' in word,
    }

    # Extract some key features from previous token potentially helpful for POS tagging
    if i > 0:
        prev = sentence[i - 1]
        features.update({
            '-1:word.lower()': prev.lower(),
            '-1:word[-3:]': prev[-3:],
            '-1:word.istitle()': prev.istitle(),
            '-1:word.isupper()': prev.isupper(),
        })
    else:
        features['BOS'] = True

    # Extract some key features from the next token potentially helpful for POS tagging
    if i < len(sentence) - 1:
        nxt = sentence[i + 1]
        features.update({
            '+1:word.lower()': nxt.lower(),
            '+1:word[-3:]': nxt[-3:],
            '+1:word.istitle()': nxt.istitle(),
            '+1:word.isupper()': nxt.isupper(),
        })
    else:
        features['EOS'] = True

    return features


def prepare_data_for_crf(X):
    X_crf = []
    for tokens in X:
        features_i = []
        for i in range(len(tokens)):
            features_i.append(word2features(tokens, i))
        X_crf.append(features_i)

    return X_crf

# Load train data
with open("data/train.jsonl", encoding="utf-8") as file:
    train_data = [json.loads(line) for line in file]

# Prepare dataset
X = [record["tokens"] for record in train_data]
X_train = prepare_data_for_crf(X)

y_train = [record["ner_tags"] for record in train_data]

crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_states=True)

crf.fit(X_train, y_train)

# Load test data
with open("data/test.jsonl", encoding="utf-8") as file:
    test_data = [json.loads(line) for line in file]

X_test = prepare_data_for_crf([record["tokens"] for record in test_data])
y_test = [record["ner_tags"] for record in test_data]

# Test the model
y_pred = crf.predict(X_test)

labels = [l for l in crf.classes_ if l != 'O']
print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3))


