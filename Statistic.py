import numpy as np
from scipy.stats import ttest_ind, rankdata, ranksums
from tabulate import tabulate


def t_student(headers_array, scores, alfa=.05):
    t_statistic = np.zeros((len(headers_array), len(headers_array)))
    p_value = np.zeros((len(headers_array), len(headers_array)))

    # Wyliczenie t_statystyki i p-value dla wszytskich par
    for i in range(len(headers_array)):
        for j in range(len(headers_array)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])
    # print(t_statistic)
    # Wyliczenie przewagi danego algorytmu
    advantage = np.zeros((len(headers_array), len(headers_array)))
    advantage[t_statistic > 0] = 1

    # Wyliczenie które algorytmy sa statystycznie różne
    significance = np.zeros((len(headers_array), len(headers_array)))
    significance[p_value <= alfa] = 1

    # Wymnożenie macieży przewag i macieży znaczności
    stat_better = significance * advantage
    return stat_better


def wilcoxon(headers_array, scores, alpha=.05):
    # Średnie wyniki dla każdego z foldów
    mean_scores = np.mean(scores, axis=2).T
    # Przypisanie rang od 1 do (liczby estymatorów) w przypadku remisów uśredniamy
    ranks = []
    for ms in mean_scores:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    # mean_ranks = np.mean(ranks, axis=0)

    # Obliczenie t-statisticy i p-value
    w_statistic = np.zeros((len(headers_array), len(headers_array)))
    p_value = np.zeros((len(headers_array), len(headers_array)))
    for i in range(len(headers_array)):
        for j in range(len(headers_array)):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

    advantage = np.zeros((len(headers_array), len(headers_array)))
    advantage[w_statistic > 0] = 1

    significance = np.zeros((len(headers_array), len(headers_array)))
    significance[p_value <= alpha] = 1

    # Wymnożenie macieży przewag i macieży znaczności
    stat_better = significance * advantage
    return stat_better
