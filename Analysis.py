import Charts
import numpy as np


#Odczyt danych do analizy
accuracy = np.load(r'Results\accuracy.npy') # 3-wymiarowa tabela z wartościami Accuracy
data_files_names = np.load(r'Results\data_files_names.npy') # Nazwy użytych baz danych
clf_names = np.load('Results\clf_names.npy') # Nazwy klasyfikatorów bazowych
wilcoxon_test = np.load(r'Results\wilcoxon.npy') # 2-wymiarowa tablica z testem Wilcoxona
t_student_test = []
for file_name in data_files_names:
    t_student_test.append(np.load(rf'Results\t_student_{file_name}.npy'))



Charts.GenerateLatexTable([accuracy], data_files_names.tolist())