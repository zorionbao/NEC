"""Annotation sensitivity for selected original endothelial discoveries."""
from pathlib import Path
import anndata as ad,numpy as np,pandas as pd
from analyze_revision import adjusted,bh
r=Path(__file__).resolve().parents[1];out=r/'new_results';a=ad.read_h5ad(r/'04_corrected_results/01_single_cell_corrected/corrected_hca_nec_atlas.h5ad');p=pd.read_csv(r/'inputs/native_posteriors.csv.gz',dtype={'donor_id':str});p=p.loc[p.lineage_model.eq('endothelial')&p.selection.eq('unselected')].copy();meta=pd.read_csv(r/'inputs/donor_metadata.csv',dtype={'donor_id':str}).sort_values('donor_id');pos=pd.Series(np.arange(a.n_obs),index=a.obs_names);counts=a.layers['counts'].tocsr()[pos.loc[p.cell_id_original].to_numpy()];genes=['CCL21','CSF2','CXCL1','VCAM1','SELE','CSF3'];rows=[];effects=[]
for state,w in [('LEC',p['probability__LEC'].to_numpy()),('Vascular EC',1-p['probability__LEC'].to_numpy())]:
 bank=[]
 for d in meta.itertuples():
  m=p.donor_id.eq(d.donor_id).to_numpy();cnt=np.asarray(counts[m].T@w[m]).ravel();lib=cnt.sum();bank.append(np.log2((cnt+.5)/(lib+1)*1e6) if lib>0 else np.full(a.n_vars,np.nan))
  for g in genes:
   j=a.var_names.get_loc(g);rows.append({'donor_id':d.donor_id,'condition':d.condition,'state':state,'gene':g,'weighted_count':cnt[j],'weighted_library':lib,'expected_cells':w[m].sum(),'log2_CPM':bank[-1][j]})
 pd.DataFrame(bank,index=meta.donor_id,columns=a.var_names).to_csv(out/('posterior_weighted_'+state.replace(' ','_')+'_all_genes.csv.gz'),compression='gzip')
table=pd.DataFrame(rows);table.to_csv(out/'posterior_weighted_endothelial_donor_values.csv',index=False)
for (state,gene),f in table.groupby(['state','gene']):
 f=f.merge(meta[['donor_id','gestational_age_weeks','age_at_surgery_days']],on='donor_id');res,_=adjusted(f.log2_CPM.to_numpy(),f.condition.eq('nec').to_numpy(),f[['gestational_age_weeks','age_at_surgery_days']].to_numpy());res.update(state=state,gene=gene,role='exploratory annotation sensitivity of genes selected from original vascular-state discoveries; not an independent discovery test',effect_scale='log2 CPM coefficient');effects.append(res)
results=pd.DataFrame(effects);results['fdr_bh']=bh(results.p_value);results['multiplicity_family']='12 post-selection gene-compartment sensitivity models';results['multiplicity_note']='BH12 is conditional diagnostic adjustment among selected effects, not an independent discovery FDR';results.to_csv(out/'posterior_weighted_endothelial_sensitivity.csv',index=False);print(results.to_string(index=False),flush=True)
