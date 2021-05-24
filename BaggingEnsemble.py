from sklearn import clone
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
import numpy as np
import statistics


class BaggingEnsemble(object):

    def __init__(self, base_estimators, type_voting='hard', random_state=None):
        self.base_estimatros = base_estimators  # Klasyfikatory bazowe
        self.type_voting = type_voting  # Rodzaj kombinacja
        self.random_state = np.random.RandomState(random_state)  # Ziarno losowości

    def ensemble_support_matrix(self, X):
        # Wyliczenie macierzy wsparcia
        probas_ = []
        for i, member_clf in enumerate(self.ensemble_):
            probas_.append(member_clf.predict_proba(X))
        return np.array(probas_)

    # Zrobienie baggingu i wyuczenie wszytskich modeli
    def fit(self, X_train, y_train):
        X_train, y_train = check_X_y(X_train, y_train)  # Sprawdzenie czy Z i y jest tego samegro rozmiaru
        self.n_features = X_train.shape[1]  # Zapisanie liczby atrybutów
        self.n_elements = X_train.shape[0]  # Zapisanie liczby instancji

        # Przechowywanie nazw klas (potrzebne do sprawdzanie czy model został nauczony)
        self.classes_ = np.unique(y_train)
        # Losowanie indeksów do podzbiorów z baggingu (min, max, (rozmiary tablicy))
        subspace = self.random_state.randint(0, self.n_elements, size=(len(self.base_estimatros), self.n_elements))
        # Wyuczenie każdego z podzbiorów innym algorytmem
        self.ensemble_ = []
        for i in range(len(self.base_estimatros)):
            self.ensemble_.append(clone(self.base_estimatros[i]).fit(X_train[subspace[i]], y_train[subspace[i]]))

    # Predykcja z wybraną metodą kombinacji
    def predict(self, X_test):
        # Sprawdzenie czy modele sa wyuczone
        check_is_fitted(self, "classes_")
        # Sprawdzenie poprawności danych
        X_test = check_array(X_test)
        # Sprawdzenie czy liczba cech się zgadza

        # noinspection PyUnresolvedReferences
        if X_test.shape[1] != self.n_features:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X_test)
        predict_results = []
        # Właściwy kod metod kombinacji
        if self.type_voting == 'hard':
            prediction_before_voting = np.zeros((len(self.ensemble_), len(X_test)))
            prediction = np.zeros(len(X_test))
            for i, member_clf in enumerate(self.ensemble_):
                prediction_before_voting[i] = member_clf.predict(X_test)
            for i in range(len(X_test)):
                prediction[i] = statistics.mode(prediction_before_voting[:, i])
            return prediction
        elif self.type_voting == 'soft_mean':
            # Wyliczenie sredniej wartosci wsparcia
            average_support = np.mean(esm, axis=0)
            # Wskazanie etykiet z największymi wartościami (średnimi)
            prediction = np.argmax(average_support, axis=1)
            return self.classes_[prediction]

        elif self.type_voting == 'soft_max':
            # Wyliczenie maksymalnej wartości wsparcia dla algorytmów
            max_support = np.max(esm, axis=0)
            # Wskazanie etykiety z największymi wartościami (maksów)
            prediction = np.argmax(max_support, axis=1)
            return self.classes_[prediction]

        elif self.type_voting == 'soft_min':
            # Wyliczenie minimalnej wartości wsparcia dla algorytmów
            min_support = np.min(esm, axis=0)
            # Wskazanie etykiety z największymi wartościami (minimów)
            prediction = np.argmax(min_support, axis=1)
            return self.classes_[prediction]
        else:
            raise Exception("Wrong combination flag")


    def ensemble_support_matrix(self, X):
        # Wyliczenie macierzy wsparcia
        probas_ = []
        for member_clf in self.ensemble_:
            probas_.append(member_clf.predict_proba(X))
        return np.array(probas_)
