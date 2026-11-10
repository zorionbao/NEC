"""Cloud export of all-cell donor expression for the revised display gene set."""
import argparse
from pathlib import Path
import anndata as ad
import numpy as np,pandas as pd
GENES=['HCAR2','CCL3','TNIP3','TNFAIP6','IL1B','CCL20','CSF2','CCL21','CXCL1','VCAM1','SELE','ICAM1','CXCL16','CXCR6','FOXP3','TIGIT','MUC2','OLFM4']
p=argparse.ArgumentParser();p.add_argument('--atlas',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();a=ad.read_h5ad(args.atlas);obs=a.obs.copy();obs['donor_id']=obs.donor_id.astype(str);counts=a.layers['counts'].tocsr();positions=[a.var_names.get_loc(g) for g in GENES];rows=[]
for donor in sorted(obs.donor_id.unique()):
 m=(obs.donor_id.eq(donor)&obs.selection.eq('unselected')).to_numpy();condition=obs.loc[m,'condition'].astype(str).unique();assert len(condition)==1;total=np.asarray(counts[m].sum(axis=0)).ravel();lib=total.sum();assert lib>0
 row={'donor_id':donor,'condition':condition[0],'captured_cells':int(m.sum()),'library_size':float(lib)}
 row.update({gene:float(np.log2(total[ix]/lib*1e6+.5)) for gene,ix in zip(GENES,positions)});rows.append(row)
d=pd.DataFrame(rows);assert len(d)==11;args.out.mkdir(parents=True,exist_ok=True);d.to_csv(args.out/'display_allcell_gene_donor_logcpm.csv',index=False);print('Exported',len(d),'donors and',len(GENES),'genes; log2(CPM + 0.5)',flush=True)
