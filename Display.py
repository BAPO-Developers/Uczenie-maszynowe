import latextable
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


def generate_latex_table(all_scores, dtn, t_s_arr):
    names = ['Datasets', 'GNB', 'kNN', 'Tree', 'Reg Log ', 'SVM', 'HB-H', 'HB-M', 'HB-I', 'HB-X']
    space_row = np.full(len(names), ' ', dtype=str)
    number_of_vals = all_scores[0].shape[0]  # 9
    number_of_data_sets = all_scores[0].shape[1]  # number of data sets
    number_of_folds = all_scores[0].shape[2]  # 5
    arr = []
    arr_mean = []
    rows = [names]
    for i in range(number_of_data_sets):
        for j in range(number_of_vals):
            for t in range(number_of_folds):
                arr.append(all_scores[0][j][i][t])
            arr_mean.append(np.mean(arr.copy()))
            arr = []
        int_arr = np.round(arr_mean, 10)  # generating U10 array for full dataset name
        str_arr = list(map(str, int_arr))  # int -> str array
        temp = np.insert(str_arr, 0, dtn[i])  # inserting dataset file name
        int_arr = np.array(temp[1:])  # selecting all apart from first
        str_arr = slicer_vectorized(int_arr, 0, 5)  # setting U5 array
        for k in range(1, len(temp)):  # switching items U10 -> U5
            temp[k] = str_arr[k - 1]
        rows.append(temp.copy())  # appending rows to final array
        rows.append(t_s_arr[i])
        rows.append(space_row)
        arr_mean = []

    table = texttable.Texttable()
    table.set_cols_align(["c"] * len(rows))
    table.set_deco(texttable.Texttable.HEADER | texttable.Texttable.VLINES)
    print("\\begin{tabular}{lccccccccccccccccc}")
    print('Tabulate Latex:')
    print(tabulate(rows, headers='firstrow', tablefmt='latex'))


def generate_single_square_latex_table(clfs_names, p_values):
    names = np.insert(clfs_names, 0, '', axis=0)
    rows = [names]
    for j in range(len(p_values)):
        latex_array = [clfs_names[j]]
        for k in range(len(p_values[j])):
            latex_array.append(p_values[j][k])
        rows.append(latex_array)

    table = texttable.Texttable()
    table.set_cols_align(["c"] * len(rows))
    table.set_deco(texttable.Texttable.HEADER | texttable.Texttable.VLINES)
    print("\\\nbegin{tabular}{lcccccccccc}")
    print('Tabulate Latex:')
    print(tabulate(rows, headers='firstrow', tablefmt='latex'))
