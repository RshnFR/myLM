from sklearn.linear_model import LogisticRegression       
from datasets import load_dataset       
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Dataset
ds = load_dataset("stanfordnlp/imdb")
train_data = ds['train']
test_data = ds['test']

# Training
vectorizer = TfidfVectorizer()
vectorizer.fit(train_data['text'])
X_train = vectorizer.transform(train_data['text'])
Y_train = train_data['label']
model = LogisticRegression()
model.fit(X_train, Y_train)

# Testing

X_test = vectorizer.transform(test_data['text'])
Y_test = test_data['label']
Y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(Y_test, Y_pred))