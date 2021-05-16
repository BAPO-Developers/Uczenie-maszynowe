import BaggingEnsemble as bg
import PrepareDataSets
import Statistic
import Charts
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np
from scipy.stats import ttest_ind
import time
import math


def add_data_sets_files():
    # data_sets.append(PrepareDataSets.PrepareDataSets('Iris2.csv', ','))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Iris.csv', ','))

    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[1] Haberman//haberman.csv', ';'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[2] Bupa//bupa.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[3] Ionosphere//ionosphere.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[4] Monk//monk-2.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[5] Phoneme//phoneme.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[6] Banana//banana.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[7] Pima//pima.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[8] Appendicitis//appendicitis.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[9] Tic-Tac-Toe//tic-tac-toe.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[10] Heart//heart.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[11] Wine//wine.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[12] Australian Credit//australian.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[13] Breast Cancer Wisconcil//wisconsin.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[14] Keppler//egzo.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[15] Magic Gamma//magic.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[16] Ringnorm//ring.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[17] South African Hearth//saheart.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[18] Titanic//titanic.csv'))
    # data_sets.append(
    #     PrepareDataSets.PrepareDataSets('Data sets//[19] Congressional Voting Records//housevotes.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[20] Credit Approval//crx.csv'))


def show_data_sets_chart():
    y = []
    z = []
    x = []
    for single_data in data_sets:
        y.append(single_data.n_atributes)
        z.append(single_data.n_instances)
        x.append(single_data.file_name)
    CH = Charts.data_sets_chart(y, z, x)


# ----------------==================--------------- CODE START HERE ----------------==================---------------
start_time = time.time()
random_state_decision_trees = 42
data_sets = []
add_data_sets_files()

base_clfs = [DecisionTreeClassifier(random_state=random_state_decision_trees), SVC(probability=True),
             KNeighborsClassifier(), GaussianNB()]
clfs = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier(random_state=random_state_decision_trees),
        bg.BaggingEnsemble(base_clfs)]
names = ['GNB', 'kNN', 'Tree (CART)', 'Hetero Bagging']
combinations = ['hard', 'soft_mean', 'soft_min', 'soft_max']
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=1234)
scores = np.zeros((len(clfs), len(data_sets), n_splits))

for data_id, single_data in enumerate(data_sets):
    # print(f'FILE: {single_data.file_name}')
    X = single_data.data.iloc[:, :-1]
    y = single_data.data.iloc[:, -1]

    for fold_id, (train, test) in enumerate(kf.split(X, y)):
        for clf_id, clf in enumerate(clfs):
            clf.fit(X.iloc[train], y.iloc[train])
            y_pred = clf.predict(X.iloc[test])
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

end_time = time.time()

print('\n___________________________________________')
print('Experiment data:\n')
print('Number of classifiers:', scores.shape[0])
print('Number of data sets:', scores.shape[1])
print('Number of folds:', scores.shape[2])
print(f'\nTotal time: {"{:.2f}".format(end_time-start_time)} [s] '
      f'\nPer iter: {"{:.2f}".format((end_time-start_time)/scores.shape[1])} [s]')
print('___________________________________________')

mean_scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", *names, sep="   ")
print(*mean_scores, sep="\n")

print(f'\n\nTest t-Studenta: \n{Statistic.t_student_for_all_files(clfs, names, scores, 0.05)}\n')
print(f'\n\nTest Wilcoxona: \n{Statistic.wilcoxon(clfs, names, scores)}\n')
# show_data_sets_chart()

# data = [x for x in data_sets if x.n == 'Iris2.csv']

# ---------------- Test metod głosowaia  ------------- #
# matched_obj = next(x for x in data_sets if x.file_name == 'monk-2') # write here file number as data
# data = matched_obj.data
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]
# clfs = [DecisionTreeClassifier(), SVC(probability=True), KNeighborsClassifier(), GaussianNB(), LogisticRegression(solver='lbfgs', max_iter=1000)]
#
# for single_data in data_sets:
#     print(f'FILE: {single_data.file_name}')
#     X = single_data.data.iloc[:, :-1]
#     y = single_data.data.iloc[:, -1]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     combinations = ['hard', 'soft_mean', 'soft_min', 'soft_max']
#     BE = BaggingEnsemble.BaggingEnsemble(clfs)
#     BE.fit(X_train, y_train)
#     for comb in combinations:
#         res = BE.predict(X_test, comb)
#         print(f'Accuracy {comb}: {accuracy_score(y_test, res)}')
#     print('---------------------------------------\n')
# # ---------------- Test metod głosowaia (end) ------------- #


# occ_dict = {}
#
# for item in res:
#     if item not in occ_dict:
#         occ_dict[item] = 1
#     else:
#         occ_dict[item] += 1
#
# print(occ_dict)


# accuracy_score(y_test, prediction)
# precision_score(y_test, prediction)
# recall_score(y_test, prediction)
# f1_score(y_test, prediction)
