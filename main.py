from sklearn import clone
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
import numpy as np
import statistics

class BaggingEnsable:
    def __init__(self, base_estimators, type_voting = 'hard', random_state=None):
        self.base_estimatros = base_estimators # Klasyfikatory bazowe
        self.type_voting = type_voting # Rodzaj kombinacja
        self.random_state = random_state # Ziarno losowości

# Zrobienie baggingu i wyuczenie wszytskich modeli
    def fit(self, X, y):
        X, y = check_X_y(X, y) # Sprawdezenie czy Z i y jest tego samegro rozmiaru
        self.n_features = X.shape[1] # Zapisanie liczby klas
        self.n_elements = X.shape[0]  # Zapisanie liczby elementow

        # Przehowywanie nazw klas (potrzebne do sprawdzanie czy model został nauczony)
        self.classes_ = np.unique(y)
        # Losowanie indeksów do podzbiorów z baggingu (min, max, (rozmiary tablicy))
        self.subspace = np.random.randint(0, self.n_elements, (len(self.base_estimatros), self.n_elements))
        # Wyuczenie każdego z podzbiorów innym algorytmem
        self.ensemble_ = []
        for i in range(len(self.base_estimatros)):
            self.ensemble_.append(clone(self.base_estimatros[i]).fit(X[self.subspace[i]], y[self.subspace[i]]))


# Predykcja z wybraną metodą kmbinacji
    def predict(self, X):
        # Sprawdzenie czy modele sa wyuczone
        check_is_fitted(self, "classes_")
        # Sprawdzenie poprawności danych
        X = check_array(X)
        # Sprawdzenie czy liczba cech się zgadza
        if X.shape[1] != self.n_features:
            raise ValueError("number of features does not match")

        # Właściwy kod metod kombinacji
        if self.type_voting == 'hard':
            # Podejmowanie decyzji na podstawie głosowania większościowego
            pred_ = []
            pred_2 = []
            for i, member_clf in enumerate(self.ensemble_):
                pred_.append(member_clf.predict(X))
                pred_2.append(member_clf)
            print(pred_)
        elif self.type_voting == 'soft_mean':
            print("Średnie")

        elif self.type_voting == 'soft_max':
            print("Max")

        elif self.type_voting == 'soft_min':
            print("Min")
        else:
            raise Exception("Wrong combination flag")
            print("Średnie")






from sklearn.tree import DecisionTreeClassifier
import pandas as pd

data = pd.read_csv('Iris.csv')

X = data.drop(columns=['Species'])
y = data['Species']

ensable = BaggingEnsable([DecisionTreeClassifier(), DecisionTreeClassifier()])
ensable.fit(X, y)
ensable.predict(X)

tab = [3 ,54, 3, 45, 6]

print(statistics.mode(tab))