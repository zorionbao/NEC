"""Cloud calculation of transparent primary-model influence diagnostics."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
from analyze_revision import adjusted
p=argparse.ArgumentParser();p.add_argument('--new-results',type=Path,required=True);args=p.parse_args();out=args.new_results
d=pd.read_csv(out/'donor_state_metrics.csv',dtype={'donor_id':str});meta=pd.read_csv(out/'donor_covariates.csv',dtype={'donor_id':str}).set_index('donor_id')
w=d.pivot(index='donor_id',columns='state',values='state_contrast');meta=meta.loc[w.index]
y=w[['M cell','LEC']].mean(axis=1).to_numpy();g=meta.condition.eq('nec').to_numpy();cov=meta[['gestational_age_weeks','age_at_surgery_days']].to_numpy(float)
donors=meta.reset_index();donors['primary_contrast']=y;donors.to_csv(out/'primary_contrast_donor_values.csv',index=False)
rows=[]
for i,donor in enumerate(w.index):
 keep=np.arange(len(y))!=i;r,_=adjusted(y[keep],g[keep],cov[keep]);r.update(omitted_donor=donor,omitted_condition=meta.iloc[i].condition);rows.append(r)
pd.DataFrame(rows).to_csv(out/'primary_contrast_leave_one_donor.csv',index=False)
x=np.column_stack([np.ones(len(y)),g]);pinv=np.linalg.pinv(x);beta=pinv@y;h=np.einsum('ij,ji->i',x,pinv);res=y-x@beta;se=np.sqrt(np.sum(pinv[1]**2*(res/(1-h))**2));crit=stats.t.ppf(.975,len(y)-2)
r={'state':'M cell + LEC','model':'Unadjusted','coefficient':beta[1],'ci_low':beta[1]-crit*se,'ci_high':beta[1]+crit*se}
a=pd.read_csv(out/'formal_contrast_adjusted.csv');a=a.loc[a.state.eq('M cell + LEC')].copy();a['model']='GA and surgery age adjusted';pd.concat([pd.DataFrame([r]),a],ignore_index=True).to_csv(out/'primary_contrast_model_comparison.csv',index=False)
print('Saved primary contrast diagnostics',flush=True)
