import BaggingEnsemble as bg
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import tqdm

datasets_names = ['twonorm', 'australian', 'chess', 'spectfheart', 'german', 'wisconsin', 'sonar', 'ring',
                  'saheart', 'titanic', 'housevotes', 'haberman', 'bupa', 'ionosphere', 'monk-2', 'phoneme',
                  'banana', 'pima', 'appendicitis', 'tic-tac-toe']
random_state_decision_trees = 42
random_state_bagging = 1410
n_splits = 5 # Liczba foldów walidacji krzyżowej

base_clfs = [DecisionTreeClassifier(random_state=random_state_decision_trees), SVC(probability=True),
             KNeighborsClassifier(), GaussianNB(), LogisticRegression(solver='lbfgs', max_iter=10000)]
clfs = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier(random_state=random_state_decision_trees),
        LogisticRegression(solver='lbfgs', max_iter=10000),
        SVC(probability=True), bg.BaggingEnsemble(base_clfs, 'hard', random_state_bagging),
        bg.BaggingEnsemble(base_clfs, 'soft_mean', random_state_bagging),
        bg.BaggingEnsemble(base_clfs, 'soft_min', random_state_bagging),
        bg.BaggingEnsemble(base_clfs, 'soft_max', random_state_bagging)]

clfs_names = ['GNB', 'kNN', 'Tree', 'Reg Log', 'SVM', 'HB Hard', 'HB Mean', 'HB Min', 'HB Max']

kf = KFold(n_splits=n_splits, shuffle=True, random_state=1234)

score_acc = np.zeros((len(clfs), len(datasets_names), n_splits))

for data_id, single_data in tqdm.tqdm(enumerate(datasets_names)):
    single_data = pd.read_csv(fr'Data_sets\{single_data}.csv', ',')
    X = single_data.iloc[:, :-1]
    y = single_data.iloc[:, -1]

    for fold_id, (train, test) in enumerate(kf.split(X, y)):
        for clf_id, clf in enumerate(clfs):
            clf.fit(X.iloc[train], y.iloc[train])
            y_pred = clf.predict(X.iloc[test])
            score_acc[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

np.save(r'Results\data_files_names.npy', np.array(datasets_names))
np.save(r'Results\accuracy.npy', score_acc)
np.save(r'Results\clf_names.npy', np.array(clfs_names))
