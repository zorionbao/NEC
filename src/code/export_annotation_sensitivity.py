"""Integer-count pseudobulks under alternative endothelial/target definitions."""
from pathlib import Path
import argparse,anndata as ad,pandas as pd,numpy as np
p=argparse.ArgumentParser();p.add_argument('--native',action='store_true');args=p.parse_args();r=Path(__file__).resolve().parents[1];out=r/'04_corrected_results/01_single_cell_corrected';a=ad.read_h5ad(out/'corrected_hca_nec_atlas.h5ad');obs=a.obs.copy();obs.donor_id=obs.donor_id.astype(str);counts=a.layers['counts'].tocsr();states={'endothelial_all':obs.cell_state.isin(['LEC','vascular_endothelial']).to_numpy()}
if args.native:
 q=pd.read_csv(r/'inputs/native_posteriors.csv.gz');positions=pd.Series(np.arange(a.n_obs),index=a.obs_names)
 for lin,target,name in [('epithelial','M cell','native_M_cell'),('endothelial','LEC','native_LEC'),('immune','Treg','native_T_reg')]:
  sub=q.loc[q.lineage_model.eq(lin)&q.scanvi_label.eq(target)];m=np.zeros(a.n_obs,bool);m[positions.loc[sub.cell_id_original.astype(str)].to_numpy()]=True;states[name]=m
 sub=q.loc[q.lineage_model.eq('endothelial')&q.scanvi_label.ne('LEC')];m=np.zeros(a.n_obs,bool);m[positions.loc[sub.cell_id_original.astype(str)].to_numpy()]=True;states['native_vascular_endothelial']=m
meta=pd.read_csv(out/'pseudobulk_metadata.csv');meta=meta.loc[~meta.cell_state.isin(states)];rows=[];folder=out/'pseudobulk_counts_by_cell_state'
for state,state_mask in states.items():
 cols={}
 for d in sorted(obs.donor_id.unique()):
  m=state_mask&obs.selection.eq('unselected').to_numpy()&obs.donor_id.eq(d).to_numpy()
  if m.sum()<10:continue
  sample='D'+d+'__'+state;cols[sample]=np.asarray(counts[m].sum(0)).ravel().astype(np.int64);rows.append({'pseudobulk_id':sample,'donor_id':d,'condition':str(obs.loc[m,'condition'].iloc[0]),'cell_state':state,'n_cells':int(m.sum())})
 if cols:pd.DataFrame({'gene':a.var_names,**cols}).to_csv(folder/f'pseudobulk_counts_{state}.csv.gz',index=False,compression='gzip')
pd.concat([meta,pd.DataFrame(rows)],ignore_index=True).to_csv(out/'pseudobulk_metadata.csv',index=False);pd.DataFrame(rows).to_csv(r/'new_results/annotation_sensitivity_pseudobulk_units.csv',index=False);print(pd.DataFrame(rows).groupby(['cell_state','condition']).size().to_string())
