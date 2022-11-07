import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.ensemble import RandomForestClassifier

col_names = ["SepalLength","SepalWidth","PetalLength","PetalWidth","Species"]
df = pd.read_csv("iris.csv",names = col_names)
#print(df.head())

#data preprocessing
label_map = {
    'Iris-setosa':0,
    'Iris-versicolor':1,
    'Iris-virginica':2
}
df['Species'].replace(label_map,inplace = True)
#print(df.head())
attributes = ["SepalLength","SepalWidth","PetalLength","PetalWidth"]
#normalising the data
for attribute in attributes :
    mean, sigma = df[attribute].mean(),df[attribute].std()
    df[attribute] = (df[attribute]-mean)/sigma

data = np.array(df)
X = data[:, :-1]
Y = data[:, -1]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 27)
print(f"no of training examples : {X_train.shape[0]}")
print(f"no of testing examples : {X_test.shape[0]}")
#linear function
linear_clf = svm.SVC(kernel = 'linear')
linear_clf = linear_clf.fit(X_train,Y_train)
Y_pred = linear_clf.predict(X_test)
print("\n"+"#"*25 + "  SVM ACCURACY  " + "#"*25+"\n")
accuracy = np.sum((Y_pred == Y_test)/len(Y_test))
print("Linear function SVM accuracy : ", accuracy)
#quadratic function
quad_clf = svm.SVC(kernel='poly')
quad_clf = quad_clf.fit(X_train, Y_train)
Y_pred = quad_clf.predict(X_test)
accuracy = np.sum((Y_pred == Y_test)/len(Y_test))
print("Quadratic function SVM accuracy : ", accuracy)
#radial basis function
radial_clf = svm.SVC(kernel='rbf')
radial_clf = radial_clf.fit(X_train, Y_train)
Y_pred = radial_clf.predict(X_test)
accuracy = np.sum((Y_pred == Y_test)/len(Y_test))
print("RBF(radial basis function) SVM accuracy : ", accuracy)
#MLP classifier
print("\n"+"#"*25 + "  MLP CLASSIFIER  " + "#"*25+"\n")
mlp_clf1 = MLPClassifier(batch_size=32, solver='lbfgs', learning_rate_init=0.001, hidden_layer_sizes=(16, ))
mlp_clf1.fit(X_train, Y_train)
y_pred = mlp_clf1.predict(X_test)
print("Accuracy of MLP classifier for 1 hidden layer and 16 nodes : ", np.sum((Y_pred == Y_test)/len(Y_test)))
mlp_clf2 = MLPClassifier(batch_size=32, solver='lbfgs', learning_rate_init=0.001, hidden_layer_sizes=(256, 16))
mlp_clf2.fit(X_train, Y_train)
y_pred = mlp_clf2.predict(X_test)
print("Accuracy of MLP classifier for 2 hidden layers - 256 and 16 nodes respectively : ", np.sum((Y_pred == Y_test)/len(Y_test)))
#learning rate vs accuracy
learn_rates = [0.1,0.01,0.001,0.0001,0.00001]
mlp_accuracies = []
for rate in learn_rates :
    mlp_clf = MLPClassifier(batch_size=32, solver='lbfgs', learning_rate_init=rate, hidden_layer_sizes=(256, 16))
    mlp_clf.fit(X_train,Y_train)
    Y_pred = mlp_clf.predict(X_test)
    mlp_accuracies.append(np.sum((Y_pred == Y_test)/len(Y_test)))
print("Learning Rates : ",learn_rates)
print("Corresponding MLP claasifier accuracy :", mlp_accuracies)
plt.plot(learn_rates, mlp_accuracies)
plt.xlabel('Learning rates')
plt.ylabel('Accuracy')
plt.title('Learning rate vs Accuracy')
plt.show()
#backward elimination
print("\n"+"#"*25 + "  BAcKWARD ELIMINATION METHOD PROCESSING  " + "#"*25 + "\n")
bfs = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1), k_features=(1, 4), forward=False, floating=False, verbose=2, scoring='accuracy', cv=5).fit(X_train, Y_train)
print("\n\n"+"#"*25 + "  BEST FEATURES  " + "#"*25 + "\n")
for t in bfs.k_feature_names_:
    print(attributes[int(t)-1])
clf_1 = svm.SVC(kernel='poly', degree=2)
clf_2 = svm.SVC(kernel='rbf', degree=2)
clf_3 = MLPClassifier(batch_size=32, solver='lbfgs', learning_rate_init=0.1, hidden_layer_sizes=(256, 16))
# ensemble learning (max voting technique)
ensemble_clf = EnsembleVoteClassifier(clfs=[clf_1, clf_2, clf_3], weights=[1, 1, 1])
ensemble_clf.fit(X_train, Y_train)
Y_pred = ensemble_clf.predict(X_test)
print("\n"+"#"*25 + "  MAX-VOTE CLASSIFIER  " + "#"*25+"\n")
print('Accuracy of Max-vote classifier :- ', np.sum((Y_pred == Y_test)/len(Y_test)), '\n\n')