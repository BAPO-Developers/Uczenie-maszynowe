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
    mean_accuracy = [float(sum(l))/len(l) for l in zip(*np.mean(score_acc, axis=2).T)]
    mean_precision = [float(sum(l))/len(l) for l in zip(*np.mean(score_prec, axis=2).T)]
    mean_recall = [float(sum(l))/len(l) for l in zip(*np.mean(score_rec, axis=2).T)]
    mean_f1 = [float(sum(l))/len(l) for l in zip(*np.mean(score_f1, axis=2).T)]
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


def GenerateLatexTable(all_scores, dtn):
    names = ['Datasets', 'GNB', 'kNN', 'Tree', 'Reg Log', 'SVM', 'HB Hard', 'HB Mean', 'HB Min', 'HB Max']
    vals1 = [1, 0.883, 0.849, 0.883, 0.869, 0.929, 0.926, 0.92,  0.883, 0.889]
    vals2 = [2, 0.883, 0.849, 0.883, 0.869, 0.929, 0.926, 0.92,  0.883, 0.889]
    number_of_vals = all_scores[0].shape[0] # 9
    number_of_data_sets = all_scores[0].shape[1] # 2
    number_of_folds = all_scores[0].shape[2] # 5
    print('vals', number_of_vals)
    print('sets', number_of_data_sets)
    print('folds', number_of_folds)
    arr = []
    arr_mean = []
    arr_sets = []
    alpha = []

    rows_2 = [names]
    for i in range(number_of_data_sets):
        for j in range(number_of_vals):
            for t in range(number_of_folds):
                arr.append(all_scores[0][j][i][t])
            arr_mean.append(np.mean(arr.copy()))
            arr = []
        arr_sets = np.round(arr_mean, 3)
        alpha = list(map(str, arr_sets))
        temp = np.insert(alpha, 0, dtn[i])
        rows_2.append(temp.copy())
        arr_sets = []
        arr_mean = []
    print(rows_2)
    # for sc_id, score in enumerate(all_scores):
    #     ll = []
    #     oo = []
    #     ii = []
    #     mean_acc = np.mean(score, axis=2).T
    #     ii = np.round(mean_acc, 3)
    #     print(ii)
    #     ll = ii[0]
    #     oo = np.insert(ll, 0, sc_id)
    #     print(oo)
    #     rows.append(oo)
    # # rows = [names, vals1, vals2]
    rows = rows_2.copy()
    print('Tabulate Table:')
    print(tabulate(rows, headers='firstrow'))
    table = texttable.Texttable()
    table.set_cols_align(["c"] * len(rows))
    table.set_deco(texttable.Texttable.HEADER | texttable.Texttable.VLINES)
    print('\nTabulate Latex:')
    print(tabulate(rows, headers='firstrow', tablefmt='latex'))
