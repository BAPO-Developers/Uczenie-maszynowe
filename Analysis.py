import Charts
import numpy as np

# Odczyt danych do analizy
accuracy = np.load(r'Results\accuracy.npy')  # 3-wymiarowa tabela z wartościami Accuracy
data_files_names = np.load(r'Results\data_files_names.npy')  # Nazwy użytych baz danych
clf_names = np.load('Results\clf_names.npy')  # Nazwy klasyfikatorów bazowych
wilcoxon_test = np.load(r'Results\wilcoxon.npy')  # 2-wymiarowa tablica z testem Wilcoxona
t_student_test = []
for file_name in data_files_names:
    t_student_test.append(np.load(rf'Results\t_student_{file_name}.npy'))


def prepare_latex_data():
    sign_t = []
    for t_s in t_student_test:
        sign_t = [""]
        for i in range(len(t_s)):
            better_than = ""
            check = []
            for j in range(len(t_s)):
                if t_s[j][i] == 1:
                    check.append(1)
                    better_than += str(j + 1)
                    better_than += ','
            if len(check) == len(clf_names) - 1:
                better_than = "all"
            if better_than != '-' and better_than != 'all':
                sign_t.append(better_than[:-1])
            else:
                sign_t.append(better_than)

        for u in range(1, len(sign_t)):
            if len(sign_t[u]) == 0 or sign_t[u] is None:
                sign_t[u] = "-"
        latex_array.append(sign_t)


latex_array = []
prepare_latex_data()

Charts.GenerateLatexTable([accuracy], data_files_names.tolist(), latex_array)
