from pathlib import Path
import anndata as ad
import numpy as np,pandas as pd,scvi,torch,json,time
root=Path(__file__).resolve().parents[1]
scvi.settings.seed=20261106
torch.set_num_threads(8)
rows=[];details=[]
for lineage in ['epithelial','endothelial','immune']:
 start=time.time();a=ad.read_h5ad(root/'legacy_analysis/results/dr_cmap'/f'{lineage}_drcmap_mapped.h5ad');q=a[a.obs.query_kind.eq('strict_nec_query')].copy();model=scvi.model.SCANVI.load(root/'legacy_analysis/results/dr_cmap/models'/f'{lineage}_scanvi',adata=a,accelerator='cpu');p=model.predict(adata=q,soft=True,batch_size=1024,use_posterior_mean=True);meta=q.obs[['cell_id_original','donor_id','sample_id','condition','selection','scanvi_label']].copy();meta.index=p.index;meta['previous_scanvi_label']=meta['scanvi_label'].astype(str);meta['scanvi_label']=p.idxmax(axis=1).astype(str);p.columns=['probability__'+str(c) for c in p.columns];frame=meta.join(p);frame['lineage_model']=lineage;target={'epithelial':'M cell','endothelial':'LEC','immune':'Treg'}[lineage];frame['target_state']=target;rows.append(frame.reset_index(drop=True));details.append({'lineage':lineage,'reference_cells':int(a.obs.query_kind.eq('reference').sum()),'query_cells':len(q),'query_depleted_cells':int(q.obs.selection.ne('unselected').sum()),'all_cells':len(a),'hard_labels_changed':int(meta.scanvi_label.ne(meta.previous_scanvi_label).sum()),'seconds':time.time()-start,'training':'pre-existing saved scANVI model; fresh cloud posterior inference'});print(details[-1],flush=True)
temporary=root/'inputs/native_posteriors.part.csv.gz'
full=pd.concat(rows,ignore_index=True)
full=full.astype({c:np.float64 for c in full if c.startswith('probability__')})
for lineage,group in full.groupby('lineage_model'):
 columns=[c for c in full if c.startswith('probability__') and group[c].notna().all()]
 matrix=group[columns].to_numpy(dtype=np.float64);matrix=matrix/matrix.sum(axis=1,keepdims=True)
 full.loc[group.index,columns]=matrix
 full.loc[group.index,'posterior_max_probability']=matrix.max(axis=1)
 full.loc[group.index,'posterior_entropy_normalized']=-np.sum(matrix*np.log(np.clip(matrix,1e-12,None)),axis=1)/np.log(matrix.shape[1])
full.to_csv(temporary,index=False,compression='gzip')
temporary.replace(root/'inputs/native_posteriors.csv.gz')
(root/'logs/native_scanvi_inference.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
