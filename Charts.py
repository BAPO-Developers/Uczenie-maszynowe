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


def slicer_vectorized(a,start,end):
    b = a.view((str,1)).reshape(len(a),-1)[:,start:end]
    return np.fromstring(b.tostring(),dtype=(str,end-start))


def GenerateLatexTable(all_scores, dtn):
    names = ['Datasets', 'GNB', 'kNN', 'Tree', 'Reg Log', 'SVM', 'HB Hard', 'HB Mean', 'HB Min', 'HB Max']
    number_of_vals = all_scores[0].shape[0] # 9
    number_of_data_sets = all_scores[0].shape[1] # 2
    number_of_folds = all_scores[0].shape[2] # 5
    print('vals', number_of_vals)
    print('sets', number_of_data_sets)
    print('folds', number_of_folds)
    arr = []
    arr_mean = []
    rows_2 = [names]
    for i in range(number_of_data_sets):
        for j in range(number_of_vals):
            for t in range(number_of_folds):
                arr.append(all_scores[0][j][i][t])
            arr_mean.append(np.mean(arr.copy()))
            arr = []
        arr_sets = np.round(arr_mean, 10)
        alpha = list(map(str, arr_sets))
        temp = np.insert(alpha, 0, dtn[i])
        newar = np.array(temp[1:])
        arrry = slicer_vectorized(newar, 0, 5)
        for k in range(1, len(temp)):
            temp[k] = arrry[k - 1]
        rows_2.append(temp.copy())
        arr_mean = []
    rows = rows_2.copy()
    # print('Tabulate Table:')
    # print(tabulate(rows, headers='firstrow'))
    table = texttable.Texttable()
    table.set_cols_align(["c"] * len(rows))
    table.set_deco(texttable.Texttable.HEADER | texttable.Texttable.VLINES)
    print('\nTabulate Latex:')
    print(tabulate(rows, headers='firstrow', tablefmt='latex'))
