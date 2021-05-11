import BaggingEnsemble
import PrepareDataSets
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split


data_sets = []
data = pd.read_csv('Iris2.csv')
PDS = PrepareDataSets.PrepareDataSets(True)

data_sets.append(PDS.add_file('Iris2.csv'))
data_sets.append(PDS.add_file('Iris.csv'))
data_sets.append(PDS.add_file('Data sets//[1] Haberman//haberman2.dat.csv'))

# data = [x for x in data_sets if x.n == 'Iris2.csv']

X = data.drop(columns=['Species'])
y = data['Species']
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
