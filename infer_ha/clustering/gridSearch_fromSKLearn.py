from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

# defining parameter range
# param_grid = {'C': [0.1, 1, 10, 100, 1000],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'kernel': ['poly']}

from random import seed
from random import randrange


def gridSearchStart(x, y, param_grid):
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 0)   #verbose option 0 to 3
    # fitting the model for grid search
    # print("x=",x)
    # print("y=",y)

    grid.fit(x, y)
    # print("fit done")
    # print best parameter after tuning
    # print(grid.best_params_)

    # # print how our model looks after hyper-parameter tuning
    # print(grid.best_estimator_)
    grid_predictions = grid.predict(x)
    # # print classification report
    # print(classification_report(y, grid_predictions))

    return grid.best_params_['C'], grid.best_params_['gamma'], grid.best_params_['coef0']

#
# x = list([randrange(-10, 11), randrange(-10, 11)] for i in range(10))
# labels = [-1, -1, -1, 1, 1, -1, 1, 1, 1, 1]
# options = '-t 0'  # linear model
# # Training Model
# param_grid = {'C': [0.1, 1, 10],
#               'gamma': [0.1, 0.01, 1],
#               'coef0': [0, 1, 0.1],
#               'kernel': ['poly']}
# print("x is ", x)
# cvalue, gamm = gridSearchStart(x,labels, param_grid)
# print("c value = ", cvalue, " and gamma =",gamm)