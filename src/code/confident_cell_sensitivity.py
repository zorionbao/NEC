"""Diagnostic hard-confidence sensitivity; no replacement of the primary model."""
from pathlib import Path
import anndata as ad,numpy as np,pandas as pd
from analyze_revision import adjusted,z,PROGRAMMES,LINEAGES
r=Path(__file__).resolve().parents[1];out=r/'new_results';a=ad.read_h5ad(r/'04_corrected_results/01_single_cell_corrected/corrected_hca_nec_atlas.h5ad');p=pd.read_csv(r/'inputs/native_posteriors.csv.gz',dtype={'donor_id':str});meta=pd.read_csv(r/'inputs/donor_metadata.csv',dtype={'donor_id':str}).sort_values('donor_id').reset_index(drop=True);counts=a.layers['counts'].tocsr();pos=pd.Series(np.arange(a.n_obs),index=a.obs_names);donor=a.obs.donor_id.astype(str);uns=a.obs.selection.eq('unselected');rows=[]
for state,lineage in LINEAGES.items():
 q=p.loc[p.lineage_model.eq(lineage)&p.selection.eq('unselected')].copy();ids=pos.loc[q.cell_id_original.astype(str)].to_numpy();geneids=a.var_names.get_indexer(PROGRAMMES[state]);assert (geneids>=0).all()
 for threshold in [.7,.9]:
  selected=q['probability__'+state].ge(threshold).to_numpy()
  for d in meta.itertuples():
   m=q.donor_id.eq(d.donor_id).to_numpy()&selected;n=int(m.sum());total=int((donor.eq(d.donor_id)&uns).sum());c=np.asarray(counts[ids[m]].sum(0)).ravel();lib=float(c.sum());eligible=n>=10 and lib>0;score=float(np.mean(np.log2((c[geneids]+.5)/(lib+1)*1e6))) if eligible else np.nan;scorelog=float(np.mean(np.log1p(c[geneids]/lib*1e6))) if eligible else np.nan;rows.append({'donor_id':d.donor_id,'condition':d.condition,'state':state,'target_probability_threshold':threshold,'n_confident_cells':n,'total_unselected_cells':total,'fraction':n/total,'logit_abundance':np.log((n+.5)/(total-n+.5)),'library_size':lib,'zero_library':lib==0,'programme_eligible':eligible,'programme_score':score,'programme_log1p_CPM':scorelog,'minimum_cells_for_programme':10,'excluded_reason':'' if eligible else ('zero library' if lib==0 else 'fewer than 10 confident cells')})
table=pd.DataFrame(rows);table.to_csv(out/'confident_cell_donor_values.csv',index=False);effects=[]
def model(frame,y):
 frame=frame.dropna(subset=[y,'gestational_age_weeks','age_at_surgery_days']);g=frame.condition.eq('nec').to_numpy();n1=int(g.sum());n0=int((~g).sum());base={'n':len(frame),'n_nec':n1,'n_control':n0}
 if min(n1,n0)<3:return dict(base,status='not estimable: fewer than 3 eligible donors in a group')
 cov=frame[['gestational_age_weeks','age_at_surgery_days']].to_numpy(float);x=np.column_stack([np.ones(len(frame)),g,cov]);rank=np.linalg.matrix_rank(x)
 if rank<4 or len(frame)<=4:return dict(base,status='not estimable: rank or degrees of freedom')
 if np.std(frame[y],ddof=1)==0:return dict(base,status='not estimable: constant outcome')
 result,_=adjusted(z(frame[y]),g,cov);return dict(result,status='estimable; diagnostic sensitivity')
for (state,threshold),frame in table.groupby(['state','target_probability_threshold']):
 frame=frame.merge(meta[['donor_id','gestational_age_weeks','age_at_surgery_days']],on='donor_id',validate='one_to_one')
 for feature in ['logit_abundance','programme_score','programme_log1p_CPM']:
  result=model(frame,feature);result.update(state=state,target_probability_threshold=threshold,outcome=feature);effects.append(result)
pd.DataFrame(effects).to_csv(out/'confident_cell_effect_sensitivity.csv',index=False)
primary=pd.read_csv(out/'donor_state_metrics.csv');primary[['donor_id','state','posterior_expected_count','weighted_library_size']].assign(zero_expected_mass=lambda f:f.posterior_expected_count.eq(0),zero_weighted_library=lambda f:f.weighted_library_size.eq(0)).to_csv(out/'primary_weighted_library_support.csv',index=False)
print(pd.DataFrame(effects).to_string(index=False),flush=True)
