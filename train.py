import os
import numpy as np
from audio_utils import load_audio
from utils.wav2vec import extract_embedding
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

X = []
y = []

print("Human samples:", len(os.listdir("data/human")))
print("AI samples:", len(os.listdir("data/ai")))

# HUMAN = 0
for file in os.listdir("data/human"):
    path = os.path.join("data/human", file)
    audio = load_audio(path)
    emb = extract_embedding(audio)
    X.append(emb)
    y.append(0)

# AI = 1
for file in os.listdir("data/ai"):
    path = os.path.join("data/ai", file)
    audio = load_audio(path)
    emb = extract_embedding(audio)
    X.append(emb)
    y.append(1)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

if len(X) < 4:
    print("Not enough samples to train")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)

if len(X_test) > 0:
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
else:
    print("Test set too small, skipping accuracy")

joblib.dump(clf, "model/classifier.joblib")
print("Model saved ✔")
