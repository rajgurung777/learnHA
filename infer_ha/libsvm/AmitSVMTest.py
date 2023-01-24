import numpy
import matplotlib.pyplot as plt
from random import seed
from random import randrange

import svmutil as svm

from tools import grid as gridSearch

# ******** See this for help:  https://github.com/cjlin1/libsvm/tree/master/python

seed(1)

# Creating Data (Dense)
x = list([randrange(-10, 11), randrange(-10, 11)] for i in range(10))
labels = [-1, -1, -1, 1, 1, -1, 1, 1, 1, 1]
options = '-t 0'  # linear model
# Training Model

print("x is ", x)

model = svm.svm_train(labels, x, options)
print("model.get_sv_indices() is ", model.get_sv_indices())
print("model.get_sv_indices() is ", model.get_sv_indices())
print("model.get_sv_coef() is ", model.get_sv_coef())
print(numpy.array(x)[numpy.array(model.get_sv_indices()) - 1].T)

pp = svm.svm_predict(labels, x, model)


# ******************************************
y, x = svm.svm_read_problem('heart_scale')
m = svm.svm_train(y[:200], x[:200], '-c 4')
svm.svm_predict(y[200:], x[200:], m)
rate, param = gridSearch.find_parameters('heart_scale', '-log2c -1,1,1 -log2g -1,1,1')
print("param is ", param)

# ******************************************
# Line Parameters
w = numpy.matmul(numpy.array(x)[numpy.array(model.get_sv_indices()) - 1].T, model.get_sv_coef())
b = -model.rho.contents.value
if model.get_labels()[1] == -1:  # No idea here but it should be done :|
    w = -w
    b = -b

print("w is ",w)
print("b is ",b)

# Plotting
plt.figure(figsize=(6, 6))
for i in model.get_sv_indices():
    plt.scatter(x[i - 1][0], x[i - 1][1], color='red', s=80)
train = numpy.array(x).T
plt.scatter(train[0], train[1], c=labels)
plt.plot([-5, 5], [-(-5 * w[0] + b) / w[1], -(5 * w[0] + b) / w[1]])
plt.xlim([-13, 13])
plt.ylim([-13, 13])
plt.show()