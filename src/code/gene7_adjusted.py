"""Exploratory covariate-adjusted seven-gene family, calculated on the cloud.

This family was defined for the revision before examining its effect estimates.
It is separate from the combined primary abundance-programme contrast.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_revision import adjusted,bh,z,N_PERM,SEED

p=argparse.ArgumentParser()
p.add_argument('--new-results',type=Path,required=True)
args=p.parse_args();out=args.new_results
genes={'M cell':['GP2','SPIB'],'LEC':['PROX1','ACKR4'],'Treg':['FOXP3','TIGIT','CXCR6']}
values=pd.read_csv(out/'target_programme_gene_donor_values.csv',dtype={'donor_id':str})
meta=pd.read_csv(out/'donor_covariates.csv',dtype={'donor_id':str}).set_index('donor_id')
rows=[]
for state,markers in genes.items():
 for gene in markers:
  d=values.loc[values.state.eq(state)&values.gene.eq(gene)].set_index('donor_id')
  assert d.index.is_unique and len(d)==11,(state,gene,len(d))
  d=d.join(meta[['gestational_age_weeks','age_at_surgery_days']],how='left')
  y=d.log2_cpm.to_numpy(float)
  if not np.isfinite(y).all() or np.std(y,ddof=1)==0:
   raise ValueError(f'Gene model not estimable: {state} {gene}')
  result,_=adjusted(z(y),d.condition.eq('nec').to_numpy(),d[['gestational_age_weeks','age_at_surgery_days']].to_numpy(float))
  result.update(state=state,gene=gene,family='seven programme genes; exploratory',
   outcome='diagnosis-blind standardized posterior-weighted donor log2 CPM',
   covariates='gestational age (weeks); age at surgery (days)',
   missing_rule='complete cases; no covariate imputation',seed=SEED,
   nonzero_donors=int((d.weighted_count>0).sum()),
   donors_weighted_count_ge1=int((d.weighted_count>=1).sum()),
   total_weighted_count=float(d.weighted_count.sum()),
   maximum_weighted_donor_count=float(d.weighted_count.max()))
  rows.append(result)
result=pd.DataFrame(rows);result['fdr_bh']=bh(result.p_value)
result.to_csv(out/'programme_gene7_adjusted.csv',index=False)
(out/'programme_gene7_adjusted_plan.json').write_text(json.dumps({'family':genes,'status':'exploratory revision analysis','primary':False,'unit':'independent donor','expected_n':11,'model':'standardized gene log2 CPM ~ diagnosis + gestational age + surgery age','interval':'HC3 95% t interval','null':'studentized Freedman-Lane reduced-model residual permutation','permutations':N_PERM,'seed':SEED,'multiplicity':'Benjamini-Hochberg across seven genes'},indent=2),encoding='utf-8')
print(result.to_string(index=False),flush=True)
