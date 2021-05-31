import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from tabulate import tabulate
import pandas as pd
import texttable


def data_sets_chart(y, z, x):
    df = pd.DataFrame(np.c_[y, z], index=x)
    df.plot.bar()
    plt.legend(["no. atributes", "no. instances"])
    plt.show()


def results_plot(names, score_acc, score_prec, score_rec, score_f1):
    # Wyświetlanie wyników
    mean_accuracy = [float(sum(l)) / len(l) for l in zip(*np.mean(score_acc, axis=2).T)]
    mean_precision = [float(sum(l)) / len(l) for l in zip(*np.mean(score_prec, axis=2).T)]
    mean_recall = [float(sum(l)) / len(l) for l in zip(*np.mean(score_rec, axis=2).T)]
    mean_f1 = [float(sum(l)) / len(l) for l in zip(*np.mean(score_f1, axis=2).T)]
    categories = names

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=mean_accuracy,
        theta=categories,
        fill='toself',
        name='Accuracy'
    ))
    fig.add_trace(go.Scatterpolar(
        r=mean_precision,
        theta=categories,
        fill='toself',
        name='Precision'
    ))
    fig.add_trace(go.Scatterpolar(
        r=mean_recall,
        theta=categories,
        fill='toself',
        name='Recall'
    ))
    fig.add_trace(go.Scatterpolar(
        r=mean_f1,
        theta=categories,
        fill='toself',
        name='F1'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.6, 1]
            )),
        showlegend=True
    )

    fig.show()


def slicer_vectorized(a, start, end):
    b = a.view((str, 1)).reshape(len(a), -1)[:, start:end]
    return np.fromstring(b.tostring(), dtype=(str, end - start))


def prepare_latex_data(t_student_test, clf_names):
    latex_array = []
    for t_s in t_student_test:
        latex_array.append(compute_better_than_2D(clf_names, t_s.T, '', True))
    return latex_array


def compute_better_than_2D(clfs_names, test, added_text, ret_type=False):
    latex_array = []
    sign_t = [added_text]
    for i in range(len(test)):
        better_than = ""
        check = []
        for j in range(len(test[i])):
            if test[j][i] == 1:
                check.append(1)
                better_than += str(j + 1) + ','
        if len(check) == len(clfs_names) - 1:
            better_than = "all"
        if better_than != '-' and better_than != 'all':
            sign_t.append(better_than[:-1])
        else:
            sign_t.append(better_than)
    for u in range(1, len(sign_t)):
        if len(sign_t[u]) == 0 or sign_t[u] is None:
            sign_t[u] = "-"
    latex_array.append(sign_t)
    if ret_type is True:
        return sign_t
    else:
        return latex_array


def generate_latex_table(all_scores, dtn, t_student_test, clf_names):
    t_s_arr = prepare_latex_data(t_student_test, clf_names)
    names = ['Datasets', 'GNB', 'kNN', 'Tree', 'Reg Log ', 'SVM', 'HB-H', 'HB-M', 'HB-I', 'HB-X']
    space_row = np.full(len(names), ' ', dtype=str)
    number_of_vals = all_scores[0].shape[0]
    number_of_data_sets = all_scores[0].shape[1]
    number_of_folds = all_scores[0].shape[2]
    arr = []
    arr_mean = []
    rows = [names]
    for i in range(number_of_data_sets):
        for j in range(number_of_vals):
            for t in range(number_of_folds):
                arr.append(all_scores[0][j][i][t])
            arr_mean.append(np.mean(arr.copy()))
            arr = []
        rows.append(prepare_means(arr_mean, dtn[i]))  # appending rows to final array
        rows.append(t_s_arr[i])
        rows.append(space_row)
        arr_mean = []

    table = texttable.Texttable()
    table.set_cols_align(["c"] * len(rows))
    table.set_deco(texttable.Texttable.HEADER | texttable.Texttable.VLINES)
    print("\\begin{tabular}{lccccccccccccccccc}")
    print('Tabulate Latex:')
    print(tabulate(rows, headers='firstrow', tablefmt='latex'))


def prepare_means(arr_mean, insert_str):
    int_arr = np.round(arr_mean, 10)  # generating U10 array for full dataset name
    str_arr = list(map(str, int_arr))  # int -> str array
    temp = np.insert(str_arr, 0, insert_str)  # inserting dataset file name
    int_arr = np.array(temp[1:])  # selecting all apart from first
    str_arr = slicer_vectorized(int_arr, 0, 5)  # setting U5 array
    for k in range(1, len(temp)):  # switching items U10 -> U5
        temp[k] = str_arr[k - 1]
    return temp


def generate_single_square_latex_table(all_scores, clfs_names, wilcoxon_test):
    wilcoxon_test = wilcoxon_test.T
    print(wilcoxon_test)
    mean_f = []
    mean_d = []
    #
    # for classifier in all_scores:
    #     for data in classifier:
    #         for fold in data:
    #             mean_f.append(np.mean(fold))
    #         mean_d.append(np.mean(mean_f))
    mean_d = all_scores
    int_arr = np.round(mean_d, 10)  # generating U10 array for full dataset name
    str_arr = list(map(str, int_arr))  # int -> str array
    temp = np.insert(str_arr, 0, 'Mean Ranks')  # inserting dataset file name
    int_arr = np.array(temp[1:])  # selecting all apart from first
    str_arr = slicer_vectorized(int_arr, 0, 5)  # setting U5 array
    for k in range(1, len(temp)):  # switching items U10 -> U5
        temp[k] = str_arr[k - 1]
    latex_array = compute_better_than_2D(clfs_names, wilcoxon_test, "Lepsze od")
    names = np.insert(clfs_names, 0, '', axis=0)
    rows = [names, prepare_means(mean_d, "Accuracy"), latex_array[0]]
    table = texttable.Texttable()
    table.set_cols_align(["c"] * len(rows))
    table.set_deco(texttable.Texttable.HEADER | texttable.Texttable.VLINES)
    print("\\\nbegin{tabular}{lcccccccccc}")
    print('Tabulate Latex:')
    print(tabulate(rows, headers='firstrow', tablefmt='latex'))

