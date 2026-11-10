"""Diagnostic sensitivity; does not change the frozen primary test."""
from pathlib import Path
import argparse,numpy as np,pandas as pd
from analyze_revision import adjusted,z,PROGRAMMES
p=argparse.ArgumentParser();p.add_argument('--new-results',type=Path,required=True);args=p.parse_args();out=args.new_results
d=pd.read_csv(out/'donor_state_metrics.csv',dtype={'donor_id':str});g=pd.read_csv(out/'target_programme_gene_donor_values.csv',dtype={'donor_id':str});meta=pd.read_csv(out/'donor_covariates.csv',dtype={'donor_id':str}).sort_values('donor_id').set_index('donor_id');g=g.merge(d[['donor_id','state','weighted_library_size','logit_abundance']],on=['donor_id','state'],validate='many_to_one');coverage=g.groupby(['state','gene']).agg(donors=('weighted_count','size'),zero_count_donors=('weighted_count',lambda v:int((v==0).sum())),donors_below_half_count=('weighted_count',lambda v:int((v<.5).sum())),minimum_weighted_count=('weighted_count','min'),median_weighted_count=('weighted_count','median'),maximum_weighted_count=('weighted_count','max')).reset_index();coverage.to_csv(out/'programme_gene_count_support.csv',index=False)
results=[];values=[]
for method in ['pseudocount_0.1','pseudocount_0.5','pseudocount_1.0','log1p_CPM']:
 f=g.copy();f['score']=np.log1p(f.weighted_count/f.weighted_library_size*1e6) if method=='log1p_CPM' else np.log2((f.weighted_count+float(method.split('_')[1]))/(f.weighted_library_size+1)*1e6);t=f.groupby(['state','donor_id']).score.mean().reset_index();t=t.merge(d[['state','donor_id','logit_abundance']],on=['state','donor_id']);t['state_contrast']=np.nan
 for state in PROGRAMMES:
  m=t.state.eq(state);t.loc[m,'state_contrast']=z(t.loc[m,'logit_abundance'])-z(t.loc[m,'score'])
 y=t.loc[t.state.isin(['M cell','LEC'])].groupby('donor_id').state_contrast.mean().loc[meta.index].to_numpy();r,_=adjusted(y,meta.condition.eq('nec').to_numpy(),meta[['gestational_age_weeks','age_at_surgery_days']].to_numpy(float));r.update(method=method,role='diagnostic sensitivity; original frozen primary retained');results.append(r);t['method']=method;values.append(t)
pd.DataFrame(results).to_csv(out/'primary_programme_transform_sensitivity.csv',index=False);pd.concat(values,ignore_index=True).to_csv(out/'programme_transform_donor_values.csv',index=False);print(pd.DataFrame(results).to_string(index=False),flush=True)
