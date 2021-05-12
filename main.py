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
    data_sets.append(PrepareDataSets.PrepareDataSets('Iris2.csv', ','))
    data_sets.append(PrepareDataSets.PrepareDataSets('Iris.csv', ','))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[1] Haberman//haberman.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[2] Bupa//bupa.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[3] Ionosphere//ionosphere.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[4] Monk//monk-2.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[5] Phoneme//phoneme.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[6] Banana//banana.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[7] Pima//pima.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[8] Appendicitis//appendicitis.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[9] Tic-Tac-Toe//tic-tac-toe.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[10] Heart//heart.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[11] Wine//wine.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[12] Australian Credit//australian.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[13] Breast Cancer Wisconcil//wisconsin.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[14] Breast Cancer//breast.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[15] Magic Gamma//magic.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[16] Ringnorm//ring.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[17] South African Hearth//saheart.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[18] Titanic//titanic.dat.csv'))
    data_sets.append(
        PrepareDataSets.PrepareDataSets('Data sets//[19] Congressional Voting Records//housevotes.dat.csv'))
    data_sets.append(PrepareDataSets.PrepareDataSets('Data sets//[20] Credit Approval//crx.dat.csv'))


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
matched_obj = next(x for x in data_sets if x.file_name == 'Iris')
data = matched_obj.data
# show_data_sets_chart()

# data = [x for x in data_sets if x.n == 'Iris2.csv']

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
BE = BaggingEnsemble.BaggingEnsemble([DecisionTreeClassifier(), SVC(), KNeighborsClassifier(), GaussianNB()])

BE.fit(X_train, y_train)
res = BE.predict(X_test)

occ_dict = {}

for item in res:
    if item not in occ_dict:
        occ_dict[item] = 1
    else:
        occ_dict[item] += 1

print(occ_dict)
print(f'Accuracy: {accuracy_score(y_test, res)}')



# accuracy_score(y_test, prediction)
# precision_score(y_test, prediction)
# recall_score(y_test, prediction)
# f1_score(y_test, prediction)
