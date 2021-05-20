import latextable as latextable

import BaggingEnsemble as bg
import PrepareDataSets
import Statistic
import Charts
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np
import time
import tqdm
from tabulate import tabulate


def add_data_sets_files():
    # data_sets.append(PrepareDataSets.PrepareDataSets('Iris2.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Iris.csv'))

    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[1] Haberman//haberman.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[2] Bupa//bupa.csv'))

    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[3] Ionosphere//ionosphere.csv'))  # THIS
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[4] Monk//monk-2.csv'))# THIS
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[5] Phoneme//phoneme.csv'))# THIS
    #data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[6] Banana//banana.csv'))# THIS

    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[7] Pima//pima.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[8] Appendicitis//appendicitis.csv'))

    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[9] Tic-Tac-Toe//tic-tac-toe.csv'))# THIS
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[10] Heart//heart.csv'))# THIS
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[12] Australian Credit//australian.csv'))# THIS
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[13] Breast Cancer Wisconcil//wisconsin.csv'))# THIS

    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[14] Keppler//egzo.csv'))

    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[15] Magic Gamma//magic.csv'))# THIS

    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[16] Ringnorm//ring.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[17] South African Hearth//saheart.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[18] Titanic//titanic.csv'))
    # data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[19] Congressional Voting Records//housevotes.csv')) # THIS


def show_data_sets_chart():
    y = []
    z = []
    x = []
    for single_data in data_sets:
        y.append(single_data.n_atributes)
        z.append(single_data.n_instances)
        x.append(single_data.file_name)
    Charts.data_sets_chart(y, z, x)


def show_experiment_data():
    print('\n_____________________________________________________________')
    print('Experiment data:\n')
    print('Number of classifiers:', score_acc.shape[0])  # acc taken - could be anything else
    print('Number of data sets:', score_acc.shape[1])
    print('Number of folds:', score_acc.shape[2])
    print(f'\nTotal time: {"{:.2f}".format(end_time-start_time)} [s] '
          f'\nPer iter: {"{:.2f}".format((end_time-start_time)/score_acc.shape[1])} [s]')
    print('_____________________________________________________________')


def show_scores():
    for sc_id, score in enumerate(accuracy):
        mean_acc = np.mean(score, axis=2).T
        print(f"\n{scores_names[sc_id]} mean scores:")
        table = tabulate(mean_acc, clfs_names)
        print(table)
        print(f'\nTest t-Studenta: \n{Statistic.t_student_for_all_files(clfs, clfs_names, score, 0.05)}')
        print(f'\nTest Wilcoxona: \n{Statistic.wilcoxon(clfs, clfs_names, score)}\n')
        print('\n_____________________________________________________________')


def show_general_scores():
    print('_____________________________________________________________')
    print('General means for each classifier:')
    print(*clfs_names, sep="                 ")
    print('Accuracy:')
    print([float(sum(l))/len(l) for l in zip(*np.mean(score_acc, axis=2).T)])


# ----------------==================--------------- CODE START HERE ----------------==================---------------

start_time = time.time()
random_state_decision_trees = 42
data_sets = []
add_data_sets_files()
n_splits = 5

base_clfs = [DecisionTreeClassifier(random_state=random_state_decision_trees), SVC(probability=True),
             KNeighborsClassifier(), GaussianNB(), LogisticRegression(solver='lbfgs', max_iter=1000)]
clfs = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier(random_state=random_state_decision_trees), LogisticRegression(solver='lbfgs', max_iter=1000), SVC(probability=True),
        bg.BaggingEnsemble(base_clfs, 'hard'), bg.BaggingEnsemble(base_clfs, 'soft_mean'), bg.BaggingEnsemble(base_clfs, 'soft_min'), bg.BaggingEnsemble(base_clfs, 'soft_max')]

clfs_names = ['GNB', 'kNN', 'Tree', 'Reg Log', 'SVM', 'HB Hard', 'HB Mean', 'HB Min', 'HB Max']
combinations = ['hard', 'soft_mean', 'soft_min', 'soft_max']
scores_names = ['Accuracy', 'Precision', 'Recall', 'F1']

kf = KFold(n_splits=n_splits, shuffle=True, random_state=1234)

score_acc = np.zeros((len(clfs), len(data_sets), n_splits))

for data_id, single_data in tqdm.tqdm(enumerate(data_sets)):
    print(f'FILE: {single_data.file_name}')
    X = single_data.data.iloc[:, :-1]
    y = single_data.data.iloc[:, -1]

    for fold_id, (train, test) in enumerate(kf.split(X, y)):
        for clf_id, clf in enumerate(clfs):
            clf.fit(X.iloc[train], y.iloc[train])
            y_pred = clf.predict(X.iloc[test])
            score_acc[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

# Nazwy plików baz danych

dtn = []
for n in data_sets:
    dtn.append(n.file_name)

np.save(r'Results\data_files_names.npy', np.array(dtn))
np.save(r'Results\accuracy.npy', score_acc)
np.save(r'Results\clf_names.npy', np.array(clfs_names))
np.save(r'Results\wilcoxon.npy', Statistic.wilcoxon(clfs, clfs_names, score_acc))
for i, file_name in enumerate(dtn):
    np.save(rf'Results\t_student_{file_name}.npy', Statistic.t_student(clfs, clfs_names, score_acc[:, i, :], 0.05, True))

end_time = time.time()

# show_experiment_data()
# show_scores()
# show_general_scores()
# show_data_sets_chart()
# Charts.results_plot(names, score_acc, score_prec, score_rec, score_f1)

