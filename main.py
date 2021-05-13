import BaggingEnsemble
import PrepareDataSets
import Charts
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split


def add_data_sets_files():
    data_sets.append(PrepareDataSets.PrepareDataSets('Iris2.csv', ',', True))
    data_sets.append(PrepareDataSets.PrepareDataSets('Iris.csv', ','))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[1] Haberman//haberman.csv', ';', True))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[2] Bupa//bupa.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[3] Ionosphere//ionosphere.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[4] Monk//monk-2.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[5] Phoneme//phoneme.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[6] Banana//banana.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[7] Pima//pima.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[8] Appendicitis//appendicitis.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[9] Tic-Tac-Toe//tic-tac-toe.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[10] Heart//heart.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[11] Wine//wine.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[12] Australian Credit//australian.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[13] Breast Cancer Wisconcil//wisconsin.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[14] Breast Cancer//breast.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[15] Magic Gamma//magic.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[16] Ringnorm//ring.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[17] South African Hearth//saheart.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[18] Titanic//titanic.csv'))
    data_sets.append(
        PrepareDataSets.PrepareDataSets('Data sets//[19] Congressional Voting Records//housevotes.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[20] Credit Approval//crx.csv'))


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


data_sets = []
# data = pd.read_csv('Iris2.csv')
add_data_sets_files()
matched_obj = next(x for x in data_sets if x.file_name == 'monk-2')
data = matched_obj.data
# show_data_sets_chart()

# data = [x for x in data_sets if x.n == 'Iris2.csv']

# ---------------- Test metod głosowaia  ------------- #
data = pd.read_csv('titanic.csv', ';')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
combinations = ['hard', 'soft_mean', 'soft_min', 'soft_max']
for comb in combinations:
    BE = BaggingEnsemble.BaggingEnsemble([DecisionTreeClassifier(), SVC(probability=True), KNeighborsClassifier(), GaussianNB()], 'soft_min')
    BE.fit(X_train, y_train)
    res = BE.predict(X_test)
    print(f'Accuracy {comb}: {accuracy_score(y_test, res)}')
# ---------------- Test metod głosowaia (end) ------------- #

occ_dict = {}

for item in res:
    if item not in occ_dict:
        occ_dict[item] = 1
    else:
        occ_dict[item] += 1

print(occ_dict)




# accuracy_score(y_test, prediction)
# precision_score(y_test, prediction)
# recall_score(y_test, prediction)
# f1_score(y_test, prediction)