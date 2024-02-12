from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from libsvm.svmutil import svm_train, svm_predict, svm_problem, svm_parameter
from libsvm.commonutil import svm_read_problem
import matplotlib.pyplot as plt
import numpy as np

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features here
y = iris.target

# Standardize the features
sc = StandardScaler()
X = sc.fit_transform(X)

# Prepare problem for libsvm
prob = svm_problem(y.tolist(), X.tolist())

# Create SVM parameters
# For example, a linear kernel with C=1
param = svm_parameter('-t 0 -c 1')

# Train a SVM model
model = svm_train(prob, param)

# Create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

# Plot the decision boundary
Z, _, _ = svm_predict([0]*len(xx.ravel()), np.c_[xx.ravel(), yy.ravel()].tolist(), model)
Z = np.array(Z).reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
colors = ['navy', 'turquoise', 'darkorange']
for color, i in zip(colors, [0, 1, 2]):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.coolwarm, edgecolor='k', s=20)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary')
plt.show()

