from sklearn import clone
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
import numpy as np
import statistics
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split

class BaggingEnsemble:
    def __init__(self, base_estimators, type_voting='hard', random_state=None):
        self.base_estimatros = base_estimators # Klasyfikatory bazowe
        self.type_voting = type_voting # Rodzaj kombinacja
        self.random_state = random_state # Ziarno losowości

# Zrobienie baggingu i wyuczenie wszytskich modeli
    def fit(self, X_train, y_train):
        X_train, y_train = check_X_y(X_train, y_train) # Sprawdzenie czy Z i y jest tego samegro rozmiaru
        self.n_features = X_train.shape[1] # Zapisanie liczby atrybutów
        self.n_elements = X_train.shape[0]  # Zapisanie liczby instancji

        # Przechowywanie nazw klas (potrzebne do sprawdzanie czy model został nauczony)
        self.classes_ = np.unique(y_train)
        # Losowanie indeksów do podzbiorów z baggingu (min, max, (rozmiary tablicy))
        self.subspace = np.random.randint(0, self.n_elements, (len(self.base_estimatros), self.n_elements))
        # print(self.subspace)
        # Wyuczenie każdego z podzbiorów innym algorytmem
        self.ensemble_ = []
        for i in range(len(self.base_estimatros)):
            self.ensemble_.append(clone(self.base_estimatros[i]).fit(X_train[self.subspace[i]], y_train[self.subspace[i]]))


# Predykcja z wybraną metodą kombinacji
    def predict(self, X_test):
        # Sprawdzenie czy modele sa wyuczone
        check_is_fitted(self, "classes_")
        # Sprawdzenie poprawności danych
        X_test = check_array(X_test)
        # Sprawdzenie czy liczba cech się zgadza
        if X_test.shape[1] != self.n_features:
            raise ValueError("number of features does not match")

        predict_results = []

        # Właściwy kod metod kombinacji
        if self.type_voting == 'hard':
            for X_element in X_test:
                # Podejmowanie decyzji na podstawie głosowania większościowego
                pred_ = []
                for member_clf in self.ensemble_:
                    pred_.append(member_clf.predict(X_element.reshape(1, -1)))
                new_pred = np.concatenate(pred_, axis=0)
                predict_results.append(statistics.mode(new_pred))

        elif self.type_voting == 'soft_mean':
            print("Średnie")

        elif self.type_voting == 'soft_max':
            print("Max")

        elif self.type_voting == 'soft_min':
            print("Min")
        else:
            raise Exception("Wrong combination flag")
            print("Średnie")
        return predict_results

data = pd.read_csv('Iris2.csv')

X = data.drop(columns=['Species'])
y = data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
ensemble = BaggingEnsemble([DecisionTreeClassifier(), SVC(), KNeighborsClassifier(), GaussianNB()])
ensemble.fit(X_train, y_train)
res = ensemble.predict(X_test)

occ_dict = {}

for item in res:
    if item not in occ_dict:
        occ_dict[item] = 1
    else:
        occ_dict[item] += 1

print(occ_dict)
print(f'Accuracy: {accuracy_score(y_test, res)}')



#
# accuracy_score(y_test, prediction)
# precision_score(y_test, prediction)
# recall_score(y_test, prediction)
# f1_score(y_test, prediction)
