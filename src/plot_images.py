import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc


def lines_ca_score(folder, score, options, comp_lbl, col, lines_style, mark_size, line_size):
    """
    It plots the lines for the clustering score.

    :param folder: folder that contains the results.
    :param score: measure to evaluate the cluster.
    :param options: compression methods evaluated.
    :param comp_lbl: labels of the compression techniques.
    :param col: color vector.
    :param lines_style: line style.
    :param mark_size: size of the marks.
    :param line_size: size of the lines.
    """
    fig = plt.figure(figsize=(10, 8))
    i = 0
    for compress_opt in options:
        x = pd.read_csv(f'{folder}/clustering_{compress_opt}_{score}.csv', index_col=0)
        x.index = x.index.astype(str)
        plt.plot(x.iloc[x.shape[0]:None:-1], color=col[i], marker="p", linestyle=lines_style[i],
                 linewidth=line_size[i], markersize=mark_size[i], label=comp_lbl[compress_opt])
        i = i + 1
    plt.ylabel(f'{score.upper()}', fontsize=25)
    plt.xlabel('Factors', fontsize=25)
    plt.legend(fontsize=18)
    plt.xticks(range(len(x)), [r'$\frac{1}{128}$', r'$\frac{1}{64}$', r'$\frac{1}{32}$', r'$\frac{1}{16}$',
                   r'$\frac{1}{8}$', r'$\frac{1}{4}$', r'$\frac{1}{2}$', r'$1$', r'$1.5$', r'$2$'], fontsize=25)
    plt.yticks(fontsize=20)
    plt.ylim((0,1))
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-clustering-{score}.png', bbox_inches='tight')
    plt.close()


def lines_compression(folder, metric='dtw'):
    """
    It plots the lines for the compression rates, scores, and processing time.

    :param folder: folder that contains the results.
    :param metric: distance measure selected.
    """
    # parameters for the plot
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{cmbright}')
    rc('font', size=25)
    rc('legend', fontsize=25)
    options = ['DP', 'TR', 'SP', 'TR_SP', 'SP_TR', 'DP_SP', 'SP_DP', 'TR_DP', 'DP_TR']
    comp_lbl = {'DP': 'DP', 'TR': 'TR', 'SP': 'SB', 'TR_SP': 'TR+SB', 'SP_TR': 'SB+TR', 'DP_SP': 'DP+SB',
                'SP_DP': 'SB+DP', 'TR_DP': 'TR+DP', 'DP_TR': 'DP+TR'}
    col = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'black', 'tab:purple', 'tab:brown',
           'tab:pink', 'tab:olive']
    lines_style = [(0, (3,1,1,1)), (0, (5, 1)), (0, (3, 5, 1, 5)), 'dotted',
                   (0, (1, 3)), 'dashdot', (0, (3, 3, 1, 3)), (0, (3, 1, 1, 1, 1, 1)), 'solid']
    mark_size = ['11', '11', '11', '9', '9', '6', '6', '3', '3']
    line_size = ['3', '3', '3', '2', '2', '1.5', '1.5', '1', '1']
    factors_str = [r'$\frac{1}{128}$', r'$\frac{1}{64}$', r'$\frac{1}{32}$', r'$\frac{1}{16}$',
                   r'$\frac{1}{8}$', r'$\frac{1}{4}$', r'$\frac{1}{2}$', r'$1$', r'$1.5$', r'$2$']

    # figure of compression rate
    i = 0
    for compress_opt in options:
        x = pd.read_csv(f'{folder}/{compress_opt}-compression_rates.csv')
        plt.plot(range(len(x.mean(axis=0))), x.mean(axis=0).iloc[x.shape[0]:None:-1], color=col[i], marker="p",
                 linestyle=lines_style[i], linewidth=line_size[i],
                 markersize=mark_size[i], label=comp_lbl[compress_opt])
        i = i + 1
    plt.ylabel(f'Average of Compression Rates', fontsize=25)
    plt.xlabel('Factors', fontsize=25)
    plt.legend(fontsize=18)
    plt.xticks(range(len(x.mean(axis=0))), [r'$\frac{1}{128}$', r'$\frac{1}{64}$', r'$\frac{1}{32}$', r'$\frac{1}{16}$',
                                            r'$\frac{1}{8}$', r'$\frac{1}{4}$', r'$\frac{1}{2}$', r'$1$', r'$1.5$',
                                            r'$2$'], fontsize=25)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-compression-rates.png', bbox_inches='tight')
    plt.close()

    # figure of the total processing time
    fig = plt.figure(figsize=(10, 8))
    i = 0
    for compress_opt in options:
        times_cl = pd.read_csv(f'{folder}/clustering_{compress_opt}_times.csv', index_col=0)
        times = pd.read_csv(f'{folder}/{metric}_{compress_opt}_times.csv')
        times = (times.sum(axis=0) + times_cl.T).T
        times_compression = pd.read_csv(f'{folder}/{compress_opt}-compression_times.csv')
        times_compression = times_compression
        times[1:] = (times[1:].T + times_compression.sum()).T.iloc[10:None:-1]
        plt.plot(times, color=col[i], marker="p", linestyle=lines_style[i],
                 linewidth=line_size[i], markersize=mark_size[i], label=comp_lbl[compress_opt])
        print(f'{compress_opt}:')
        print(100-(times.iloc[1:,:]/times.iloc[0,:]*100).mean())
        i = i + 1
    plt.ylabel('Processing Time (s)', fontsize=25)
    plt.xlabel('Factors', fontsize=25)
    plt.legend(fontsize=18)
    plt.xticks(range(len(times)), [r'Control'] + factors_str, fontsize=25)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-total-times.png', bbox_inches='tight')
    plt.close()

    # figure of the clustering purity
    lines_ca_score(folder, 'mh', options, comp_lbl, col, lines_style=lines_style,
                   mark_size=mark_size, line_size=line_size)
    lines_ca_score(folder, 'nmi', options, comp_lbl, col, lines_style=lines_style,
                   mark_size=mark_size, line_size=line_size)

    # figure of the pearson correlation
    fig = plt.figure(figsize=(10, 8))
    i = 0
    for compress_opt in options:
        measure = pd.read_csv(f'{folder}/measures_{metric}_{compress_opt}_times.csv', index_col=0)
        measure = measure.loc['mantel-corr']
        plt.plot(measure.iloc[measure.shape[0]:None:-1], color=col[i], marker="p", linestyle=lines_style[i],
                 linewidth=line_size[i], markersize=mark_size[i], label=comp_lbl[compress_opt])
        i = i + 1
    plt.ylabel('Mantel Correlation - Pearson', fontsize=25)
    plt.xlabel('Factors', fontsize=25)
    plt.legend(fontsize=18)
    plt.xticks(range(len(measure)), factors_str, fontsize=25)
    plt.yticks(fontsize=20)
    plt.ylim((0, 1))
    # plt.tight_layout()
    plt.savefig(f'{folder}/lines-measure-mantel.png', bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(10, 8))
    i = 0
    for compress_opt in options:
        measure = pd.read_csv(f'{folder}/measures_{metric}_{compress_opt}_times.csv', index_col=0)
        measure = measure.loc['mantel-pvalue']
        plt.plot(measure.iloc[measure.shape[0]:None:-1], color=col[i], marker="p", linestyle=lines_style[i],
                 linewidth=line_size[i], markersize=mark_size[i], label=comp_lbl[compress_opt])
        i = i + 1
    plt.ylabel('Mantel Test p-value', fontsize=25)
    plt.xlabel('Factors', fontsize=25)
    plt.legend(fontsize=18)
    plt.xticks(range(len(measure)), factors_str, fontsize=25)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-measure-mantel-pvalue.png', bbox_inches='tight')
    plt.close()

