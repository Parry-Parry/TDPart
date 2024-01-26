from fire import Fire
import pandas as pd 
from os.path import join

STAGES = ['bm25', 'splade', 'monot5']

def main(in_path : str):
    # code for latex tabular 

    preamble = r"""\begin{tabular}{lrrrrrrrr}
\toprule
& & \multicolumn{4}{c}{\textbf{BM25}} & \multicolumn{4}{c}{\textbf{SPLADEv2}} & \multicolumn{4}{c}{\textbf{monoT5}} \\
Algo & Model & nDCG@1 & nDCG@5 & nDCG@10 & P@10 & nDCG@1 & nDCG@5 & nDCG@10 & P@10 & nDCG@1 & nDCG@5 & nDCG@10 & P@10 \\
\midrule
"""

    file = pd.read_csv(in_path, sep='\t', index_col=None)

    # get the unique models
    models = file['model'].unique()
    algos = file['algo'].unique()

    lines = []

    for model in models:
        lines.append(r"\midrule")
        for algo in algos:
            line = f"{algo} & {model} & "
            for stage in STAGES:
                row = file[(file['model'] == model) & (file['algo'] == algo) & (file['stage'] == stage)]
                if len(row) == 0:
                    line += "& & & & "
                else:
                    line += f"{row['nDCG@1'].values[0]:.4f} & {row['nDCG@5'].values[0]:.4f} & {row['nDCG@10'].values[0]:.4f} & {row['P@10'].values[0]:.4f} & "
            line = line[:-3] + r"\\"
            lines.append(line)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    total = preamble + '\n'.join(lines)

    with open('main.tex', 'w') as f:
        f.write(total)

if __name__ == '__main__':
    Fire(main)