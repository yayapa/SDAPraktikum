"""
Baseline for SemEval Task{4, 5, 8}
Uses train.json and test.json files in /CNN/data/
"""
from CNN.config import Config
from CNN.utils import WordEmbeddingLoader, RelationLoader, SemEvalDataLoader
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
import numpy as np
config = Config()
config.batch_size = 1
config.embedding_path = "./CNN/embedding/hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt"
config.data_dir = "./CNN/data/"
word2id, word_vec = WordEmbeddingLoader(config).load_embedding()
rel2id, id2rel, class_num = RelationLoader(config).get_relation()
loader = SemEvalDataLoader(rel2id, word2id, config)
test_loader = loader.get_test()
train_loader = loader.get_train()

# upload train and test from dataloader
min_v, max_v = float('inf'), -float('inf')
X_train = []
y_train = []
for step, (data, label) in enumerate(train_loader):
    x = data.detach().numpy().flatten()
    x.astype(int)
    X_train.append(x)
    y_train.append(label.detach().numpy()[0])
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = []
y_test = []
for step, (data, label) in enumerate(test_loader):
    x = data.detach().numpy().flatten()
    x.astype(int)
    X_test.append(x)
    y_el = label.detach().numpy()[0]
    y_test.append(y_el)
X_test = np.array(X_test)
y_test = np.array(y_test)

# train baseline and calculate metrics
"""
print(">> SVM classifier....")
SVM = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto')
SVM.fit(X_train, y_train)
predictions_SVM = SVM.predict(X_test)
print("SVM Accuracy Score -> ", accuracy_score(y_test, predictions_SVM) * 100)
print("SVM Precision Score -> ", precision_score(y_test, predictions_SVM, average='weighted') * 100)
print("SVM Recall Score -> ", recall_score(y_test, predictions_SVM, average='weighted') * 100)
print("SVM F1 Score -> ", f1_score(y_test, predictions_SVM, average='weighted') * 100)

"""

print(">> RFC classifier....")
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
predictions_rfc = rfc.predict(X_test)
print("RFC Accuracy Score -> ", accuracy_score(y_test, predictions_rfc) * 100)
print("RFC Precision Score -> ", precision_score(y_test, predictions_rfc, average='weighted') * 100)
print("RFC Recall Score -> ", recall_score(y_test, predictions_rfc, average='weighted') * 100)
print("RFC F1 Score -> ", f1_score(y_test, predictions_rfc, average='weighted') * 100)

"""
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs', random_state=42, multi_class='multinomial', max_iter=200, n_jobs=16)
logreg.fit(X_train, y_train)
predictions_logreg = logreg.predict(X_test)
print("RFC Accuracy Score -> ", accuracy_score(y_test, predictions_logreg) * 100)
print("LogisticRegression Precision Score -> ", precision_score(y_test, predictions_logreg, average='micro') * 100)
print("LogisticRegression Recall Score -> ", recall_score(y_test, predictions_logreg, average='micro') * 100)
print("LogisticRegression F1 Score -> ", f1_score(y_test, predictions_logreg, average='micro') * 100)
"""
"""
from sklearn.tree import DecisionTreeClassifier
rfc = DecisionTreeClassifier(random_state=42)
rfc.fit(X_train, y_train)
predictions_rfc = rfc.predict(X_test)
print("DecisionTreeClassifier Accuracy Score -> ", accuracy_score(y_test, predictions_rfc) * 100)
print("DecisionTreeClassifier Precision Score -> ", precision_score(y_test, predictions_rfc, average='weighted') * 100)
print("DecisionTreeClassifier Recall Score -> ", recall_score(y_test, predictions_rfc, average='weighted') * 100)
print("DecisionTreeClassifier F1 Score -> ", f1_score(y_test, predictions_rfc, average='weighted') * 100)
"""
