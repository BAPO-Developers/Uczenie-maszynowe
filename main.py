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
import plotly.graph_objects as go
import plotly.offline as pyo


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
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[12] Australian Credit//australian.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[13] Breast Cancer Wisconcil//wisconsin.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[14] Keppler//egzo.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[15] Magic Gamma//magic.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[16] Ringnorm//ring.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[17] South African Hearth//saheart.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[18] Titanic//titanic.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[19] Congressional Voting Records//housevotes.csv'))


def show_data_sets_chart():
    y = []
    z = []
    x = []
    for single_data in data_sets:
        y.append(single_data.n_atributes)
        z.append(single_data.n_instances)
        x.append(single_data.file_name)
    CH = Charts.data_sets_chart(y, z, x)



def compute_scores():
    score_acc[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)
    score_prec[clf_id, data_id, fold_id] = precision_score(y[test], y_pred)
    score_rec[clf_id, data_id, fold_id] = recall_score(y[test], y_pred)
    score_f1[clf_id, data_id, fold_id] = f1_score(y[test], y_pred)


def show_experiment_data():
    print('\n___________________________________________')
    print('Experiment data:\n')
    print('Number of classifiers:', score_acc.shape[0])  # acc taken - could be anything else
    print('Number of data sets:', score_acc.shape[1])
    print('Number of folds:', score_acc.shape[2])
    print(f'\nTotal time: {"{:.2f}".format(end_time-start_time)} [s] '
          f'\nPer iter: {"{:.2f}".format((end_time-start_time)/score_acc.shape[1])} [s]')
    print('___________________________________________')



def show_scores():
    # -------------- ACCURACY -----------------
    mean_acc = np.mean(score_acc, axis=2).T
    print("\nAccuracy mean scores:")
    print(*names, sep="   ")
    print(*mean_acc, sep="\n")
    print('\n___________________________________________')

    # -------------- PRECISION -----------------
    mean_prec = np.mean(score_prec, axis=2).T
    print("\nPrecision mean scores:")
    print(*names, sep="   ")
    print(*mean_prec, sep="\n")
    print('\n___________________________________________')

    # -------------- RECALL -----------------
    mean_rec = np.mean(score_rec, axis=2).T
    print("\nRecall mean scores:")
    print(*names, sep="   ")
    print(*mean_rec, sep="\n")
    print('\n___________________________________________')

    # -------------- F1 -----------------
    mean_f1 = np.mean(score_f1, axis=2).T
    print("\nF1 mean scores:")
    print(*names, sep="   ")
    print(*mean_f1, sep="\n")
    print('\n___________________________________________')


def show_general_scores():
    print('___________________________________________')
    print('General means for each classifier:')
    print(*names, sep="                 ")
    print('Accuracy:')
    print([float(sum(l))/len(l) for l in zip(*np.mean(score_acc, axis=2).T)])
    print('Precision:')
    print([float(sum(l))/len(l) for l in zip(*np.mean(score_prec, axis=2).T)])
    print('Recall:')
    print([float(sum(l))/len(l) for l in zip(*np.mean(score_rec, axis=2).T)])
    print('F1:')
    print([float(sum(l))/len(l) for l in zip(*np.mean(score_f1, axis=2).T)])


def show_statistics():
    for score in all_scores:
        print(f'\n\nTest t-Studenta: \n{Statistic.t_student_for_all_files(clfs, names, score, 0.05)}\n')
        print(f'\n\nTest Wilcoxona: \n{Statistic.wilcoxon(clfs, names, score)}\n')

# ----------------==================--------------- CODE START HERE ----------------==================---------------


start_time = time.time()
random_state_decision_trees = 42
data_sets = []
add_data_sets_files()
n_splits = 5

base_clfs = [DecisionTreeClassifier(random_state=random_state_decision_trees), SVC(probability=True),
             KNeighborsClassifier(), GaussianNB()]
clfs = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier(random_state=random_state_decision_trees),
        bg.BaggingEnsemble(base_clfs)]

names = ['GNB', 'kNN', 'Tree (CART)', 'Hetero Bagging']
combinations = ['hard', 'soft_mean', 'soft_min', 'soft_max']
scores_names = ['accuracy', 'precision', 'recall', 'f1']

kf = KFold(n_splits=n_splits, shuffle=True, random_state=1234)

score_acc = np.zeros((len(clfs), len(data_sets), n_splits))
score_prec = np.zeros((len(clfs), len(data_sets), n_splits))
score_rec = np.zeros((len(clfs), len(data_sets), n_splits))
score_f1 = np.zeros((len(clfs), len(data_sets), n_splits))
all_scores = [score_acc, score_prec, score_rec, score_f1]

for data_id, single_data in enumerate(data_sets):
    print(f'FILE: {single_data.file_name}')
    X = single_data.data.iloc[:, :-1]
    y = single_data.data.iloc[:, -1]

    for fold_id, (train, test) in enumerate(kf.split(X, y)):
        for clf_id, clf in enumerate(clfs):
            clf.fit(X.iloc[train], y.iloc[train])
            y_pred = clf.predict(X.iloc[test])
            compute_scores()

end_time = time.time()

show_experiment_data()
show_scores()
show_general_scores()
# show_statistics()
# show_data_sets_chart()
Charts.results_plot(names, score_acc, score_prec, score_rec, score_f1)

