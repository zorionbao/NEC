"""Calculate the organoid gene trends used to select transport features.

The calculation is extracted from the original organoid validation analysis;
it requires only the public GSE275771 normalized-count matrix.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from revision_stats import bh_fdr


def stage_from_name(name):
    for suffix, stage in [('_PPP', 3), ('_PP', 2), ('_P', 1), ('_N', 0)]:
        if suffix in name:
            return stage
    raise ValueError(f'Unrecognized organoid sample: {name}')


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'legacy_analysis/data/public/dr_cmap/GSE275771_P11N_M_cell_bulk_seq_NormalizedCounts.csv.gz'
    output = root/'legacy_analysis/results/fxr_ferroptosis_ilc3/organoid_mcell_gene_trends.parquet'
    raw = pd.read_csv(source, index_col=0)
    raw.index = raw.index.astype(str)
    raw = raw.groupby(level=0).mean()
    expression = np.log2(raw.T+1)
    stage = np.asarray([stage_from_name(str(x)) for x in expression.index],float)
    gene_matrix = expression.T
    stage_centered = stage-stage.mean()
    centered = gene_matrix.sub(gene_matrix.mean(axis=1),axis=0)
    slopes = centered.to_numpy()@stage_centered/np.sum(stage_centered**2)
    ranks = gene_matrix.rank(axis=1,method='average')
    ranked_stage = stats.rankdata(stage,method='average')
    ranked_stage -= ranked_stage.mean()
    ranked_centered = ranks.sub(ranks.mean(axis=1),axis=0)
    numerator = ranked_centered.to_numpy()@ranked_stage
    denominator = np.sqrt(np.sum(ranked_centered.to_numpy()**2,axis=1)*np.sum(ranked_stage**2))
    rho = np.divide(numerator,denominator,out=np.full_like(numerator,np.nan,dtype=float),where=denominator>0)
    df = len(stage)-2
    t = rho*np.sqrt(df/np.maximum(1-rho**2,1e-12))
    p = 2*stats.t.sf(np.abs(t),df)
    result = pd.DataFrame({'gene':gene_matrix.index,'slope_per_stage':slopes,'spearman_rho':rho,'asymptotic_pvalue':p})
    result['fdr_bh'] = bh_fdr(p)
    output.parent.mkdir(parents=True,exist_ok=True)
    result.to_parquet(output,index=False)


if __name__ == '__main__':
    main()
