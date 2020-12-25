import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORIGIN_LAMBDA = 1.4667
MAX_HKL = 8
RESULTS_PATH = 'results'

element_df = pd.DataFrame(np.array([
    [1, 'V', 'bcc', 3.02],
    [2, 'Mo', 'bcc', 3.15],
    [3, 'Nb', 'bcc', 3.30],
    [4, 'Pt', 'fcc', 3.92],
    [5, 'Al', 'fcc', 4.09],
    [6, 'Ni', 'fcc', 3.52],
    [0, 'Pd', 'fcc', 3.89]
]), columns=['num', 'element', 'structure', 'network constant'])
element_df['num'] = element_df['num'].astype(int)
element_df['network constant'] = element_df['network constant'].astype(float)


def get_wulf_bragg_2theta(d_hkl, source_lambda):
    return np.degrees(2 * np.arcsin((1 / d_hkl) * source_lambda / 2))


def generate_HKL_DF(max_hkl):
    h_l, k_l, l_l = [], [], []
    for h in range(max_hkl + 1):
        for k in range(max_hkl + 1):
            for l in range(max_hkl + 1):
                if h + k + l != 0:
                    h_l.append(h)
                    k_l.append(k)
                    l_l.append(l)
    return pd.DataFrame({
        'h': h_l,
        'k': k_l,
        'l': l_l
    })


def get_d_hkl(a: float, h: int, k: int, l: int) -> float:
    return a / np.sqrt(h * h + k * k + l * l)


def get_bcc_pos():
    return [
        [0, 0, 0],
        [0.5, 0.5, 0.5]
    ]


def get_fcc_pos():
    return [
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
    ]


def calculate_F(h, k, l, struct_type):
    atoms_pos = []
    if struct_type == 'bcc':
        atoms_pos = get_bcc_pos()
    elif struct_type == 'fcc':
        atoms_pos = get_fcc_pos()
    else:
        raise Exception('Bad struct type')
    i = np.complex(0, 1)
    return np.sum([np.real(np.exp(2 * np.pi * i * (h * x + k * y + l * z))) for x, y, z in atoms_pos])


if __name__ == '__main__':
    album = int(sys.argv[1])
    if sys.argv[2]:
        MAX_HKL = int(sys.argv[2])
    set_num = album % 7
    particle_record = element_df[element_df['num'] == set_num]
    dyf_df = generate_HKL_DF(MAX_HKL)
    dyf_df['d_hkl'] = dyf_df.apply(lambda row: get_d_hkl(
        a=particle_record['network constant'].values[0],
        h=row['h'],
        k=row['k'],
        l=row['l']
    ), axis=1)
    dyf_df['2theta'] = dyf_df.apply(lambda row: get_wulf_bragg_2theta(row['d_hkl'], ORIGIN_LAMBDA), axis=1)
    dyf_df['F'] = dyf_df.apply(lambda row:
                               calculate_F(row['h'], row['k'], row['l'], particle_record['structure'].values[0]),
                               axis=1)
    dyf_df['visible'] = dyf_df.apply(lambda row: row['F'] > 0, axis=1)
    dyf_df = dyf_df[dyf_df['2theta'] <= 90].sort_values(by='2theta')
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
    dyf_df.to_latex(os.path.join(RESULTS_PATH, f'table_{album}.tex'), index=False)
    dyf_files = os.listdir('data')
    fig, axs = plt.subplots(figsize=(6, 10), nrows=len(dyf_files), sharex=True)
    for i, f in enumerate(dyf_files):
        with open(os.path.join('data', f)) as df:
            plt_data = pd.read_table(df, header=None, delim_whitespace=True).to_numpy()
            axs[i].plot(plt_data[:, 0], plt_data[:, 1], '-', linewidth=1)
            for x in dyf_df[dyf_df['visible']]['2theta'].unique():
                axs[i].axvline(x, color='r', alpha=0.5, linestyle=':')
            axs[i].set_title(f)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(RESULTS_PATH, f'dyf_{album}.svg'))
    plt.show()
