import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo


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
                range=[0.7, 0.9]
            )),
        showlegend=True
    )

    fig.show()
