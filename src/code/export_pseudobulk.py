from pathlib import Path
import anndata as ad,numpy as np,pandas as pd
r=Path(__file__).resolve().parents[1];out=r/'04_corrected_results/01_single_cell_corrected';a=ad.read_h5ad(out/'corrected_hca_nec_atlas.h5ad');obs=a.obs.copy();counts=a.layers['counts'].tocsr();uns=obs.selection.eq('unselected').to_numpy();di=obs.donor_id.astype(str).to_numpy();donors=sorted(set(di[uns]));meta=[];states=['all_cells']+sorted(obs.cell_state.astype(str).unique());sd=out/'pseudobulk_counts_by_cell_state';sd.mkdir(exist_ok=True)
for state in states:
 cols={}
 for d in donors:
  m=uns&(di==d)
  if state!='all_cells':m=m&obs.cell_state.eq(state).to_numpy()
  if state!='all_cells' and m.sum()<10:continue
  sample='D'+d+('__'+state if state!='all_cells' else '');cols[sample]=np.asarray(counts[m].sum(0)).ravel().astype(np.int64);meta.append({'pseudobulk_id':sample,'donor_id':d,'condition':str(obs.loc[m,'condition'].iloc[0]),'cell_state':state,'n_cells':int(m.sum())})
 if cols:
  path=(out/'pseudobulk_counts_all_cells.csv.gz') if state=='all_cells' else sd/f'pseudobulk_counts_{state}.csv.gz';pd.DataFrame({'gene':a.var_names,**cols}).to_csv(path,index=False,compression='gzip')
pd.DataFrame(meta).to_csv(out/'pseudobulk_metadata.csv',index=False)
print('EXPORTED',len(meta),'independent donor-state pseudobulks',flush=True)
