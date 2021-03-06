import Display
import numpy as np
import Statistic

# Odczyt danych do analizy
accuracy = np.load(r'Results\accuracy.npy')  # 3-wymiarowa tabela z wartościami Accuracy
data_files_names = np.load(r'Results\data_files_names.npy')  # Nazwy użytych baz danych
clf_names = np.load('Results\clf_names.npy')  # Nazwy klasyfikatorów bazowych

# 2-wymiarowa tablica z testem Wilcoxona (poziom ufności 0.05)
wilcoxon_test, mean_ranks = Statistic.wilcoxon(clf_names.tolist(), accuracy, 0.05)
# Test t-Studenta przeprowadzony dla wszytskich zbiorów danych (poziomu ufności 0.05)
t_student_test = []
print(mean_ranks)
for i, file_name in enumerate(data_files_names):
    t_student_test.append(Statistic.t_student(clf_names.tolist(), accuracy[:, i, :], 0.05))

Display.generate_latex_table([accuracy], data_files_names.tolist(), t_student_test, clf_names)
Display.generate_single_square_latex_table(mean_ranks, clf_names, wilcoxon_test)

