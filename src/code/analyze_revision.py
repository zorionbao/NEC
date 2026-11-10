"""Frozen donor-level revision analyses executed on the cloud."""
from pathlib import Path
import argparse,itertools,json,platform,time
import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
SEED=20261106
N_PERM=19999
PROGRAMMES={'M cell':['GP2','SPIB'],'LEC':['PROX1','ACKR4'],'Treg':['FOXP3','TIGIT','CXCR6']}
LINEAGES={'M cell':'epithelial','LEC':'endothelial','Treg':'immune'}
HOLDOUT_MARKERS=set(sum(PROGRAMMES.values(),[])+['SOX8','TNFRSF11A','CCL20','MARCKSL1','LYVE1','PDPN','FLT4','CCL21','IL2RA','CTLA4','IKZF2','CCR7'])
def bh(values):
 x=np.asarray(values,float);out=np.full(x.shape,np.nan);ok=np.isfinite(x);v=x[ok];order=np.argsort(v);s=np.minimum.accumulate((v[order]*len(v)/np.arange(1,len(v)+1))[::-1])[::-1];r=np.empty(len(v));r[order]=np.minimum(s,1);out[ok]=r;return out
def z(x):
 x=np.asarray(x,float);return (x-np.mean(x))/np.std(x,ddof=1)
def hedges(v,g):
 a=v[g];b=v[~g];df=len(v)-2;s=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/df);return (a.mean()-b.mean())/s*(1-3/(4*df-1)) if s else np.nan
def exact(v,g):
 v=np.asarray(v,float);g=np.asarray(g,bool);obs=float(v[g].mean()-v[~g].mean());vals=[]
 for ids in itertools.combinations(range(len(v)),g.sum()):
  mask=np.zeros(len(v),bool);mask[list(ids)]=1;vals.append(float(v[mask].mean()-v[~mask].mean()))
 return {'difference_nec_control':obs,'hedges_g':hedges(v,g),'p_value':float(np.mean(np.abs(vals)>=abs(obs)-1e-12)),'permutations':len(vals)}
def adjusted(y,group,covariates):
 y=np.asarray(y,float);g=np.asarray(group,float);c=np.asarray(covariates,float);keep=np.isfinite(y)&np.isfinite(c).all(1);y=y[keep];g=g[keep];c=c[keep];c=(c-c.mean(0))/c.std(0,ddof=1);x0=np.column_stack([np.ones(len(y)),c]);x=np.column_stack([np.ones(len(y)),g,c]);pinv=np.linalg.pinv(x);h=np.einsum('ij,ji->i',x,pinv);w=pinv[1];beta=pinv@y;res=y-x@beta;se=np.sqrt(np.sum(w*w*(res/(1-h))**2));t=beta[1]/se;fitted0=x0@np.linalg.lstsq(x0,y,rcond=None)[0];r0=y-fitted0;rng=np.random.default_rng(SEED);inds=np.array([rng.permutation(len(y)) for _ in range(N_PERM)]).T;yp=fitted0[:,None]+r0[inds];bp=pinv@yp;rp=yp-x@bp;sp=np.sqrt(np.sum(w[:,None]**2*(rp/(1-h[:,None]))**2,axis=0));tp=bp[1]/sp;p=(1+np.sum(np.abs(tp)>=abs(t)-1e-12))/(N_PERM+1);df=len(y)-np.linalg.matrix_rank(x);crit=stats.t.ppf(.975,df)
 return {'coefficient':float(beta[1]),'ci_low':float(beta[1]-crit*se),'ci_high':float(beta[1]+crit*se),'hc3_se':float(se),'t_hc3':float(t),'p_value':float(p),'n':len(y),'n_nec':int(g.sum()),'n_control':int(len(g)-g.sum()),'df':int(df),'rank':int(np.linalg.matrix_rank(x)),'condition_number':float(np.linalg.cond(x)),'max_leverage':float(h.max()),'permutations':N_PERM,'test':'studentized Freedman-Lane residual permutation (approximate)','interval':'HC3 95% t interval'},tp
def weighted_tables(a,p,meta,out):
 obs=a.obs.copy();obs.index=obs.index.astype(str);obs['donor_id']=obs.donor_id.astype(str);donors=meta.copy();donors['donor_id']=donors.donor_id.astype(str);donors=donors.sort_values('donor_id').reset_index(drop=True);uns=obs.selection.eq('unselected').to_numpy();counts=a.layers['counts'].tocsr();pos=pd.Series(np.arange(a.n_obs),index=obs.index);rows=[];gene_rows=[];bank={};marker_rows=[];uncertainty=[]
 for state,lineage in LINEAGES.items():
  q=p.loc[p.lineage_model.eq(lineage)&p.selection.eq('unselected')].copy();q.donor_id=q.donor_id.astype(str);ids=pos.loc[q.cell_id_original.astype(str)].to_numpy();w=q['probability__'+state].to_numpy(float);hard=q.scanvi_label.eq(state).to_numpy();probs=q[[c for c in q if c.startswith('probability__') and q[c].notna().all()]].to_numpy(float);assert np.allclose(probs.sum(1),1,atol=1e-5);entropy=-np.sum(probs*np.log(np.clip(probs,1e-12,None)),axis=1)/np.log(probs.shape[1])
  for condition in ['control','nec']:
   m=q.condition.eq(condition).to_numpy();uncertainty.append({'state':state,'condition':condition,'n_cells':int(m.sum()),'median_max_probability':float(np.median(probs[m].max(1))),'median_normalized_entropy':float(np.median(entropy[m])),'fraction_confident_0_7':float(np.mean(probs[m].max(1)>=.7))})
  statecounts=[]
  for d in donors.itertuples():
   m=q.donor_id.eq(d.donor_id).to_numpy();di=obs.donor_id.eq(d.donor_id).to_numpy()&uns;n=int(di.sum());sc=counts[ids[m]];weights=w[m];cnt=np.asarray(sc.T@weights).ravel();lib=cnt.sum();logcpm=np.log2((cnt+.5)/(lib+1)*1e6);abundance=float(weights.sum());logit=np.log((abundance+.5)/(n-abundance+.5));gp=[a.var_names.get_loc(g) for g in PROGRAMMES[state]];programme=float(logcpm[gp].mean());rows.append({'donor_id':d.donor_id,'condition':d.condition,'state':state,'lineage':lineage,'total_unselected_cells':n,'posterior_expected_count':abundance,'posterior_fraction':abundance/n,'abundance_fraction':abundance/n,'logit_abundance':logit,'programme_score':programme,'programme_genes':';'.join(PROGRAMMES[state]),'hard_count':int(hard[m].sum()),'confident_count':int((hard[m]&(probs[m].max(1)>=.7)).sum()),'weighted_library_size':lib})
   for g,ix in zip(PROGRAMMES[state],gp):gene_rows.append({'donor_id':d.donor_id,'condition':d.condition,'state':state,'gene':g,'log2_cpm':float(logcpm[ix]),'weighted_count':float(cnt[ix])})
   statecounts.append(logcpm)
  bank[state]=np.vstack(statecounts);pd.DataFrame(bank[state],index=donors.donor_id,columns=a.var_names).to_csv(out/f'{lineage}_weighted_donor_logcpm.csv.gz',compression='gzip')
  for g in sorted(HOLDOUT_MARKERS&set(a.var_names)):
   ix=a.var_names.get_loc(g)
   for condition in ['control','nec']:
    m=q.condition.eq(condition).to_numpy();v=counts[ids[m],ix].toarray().ravel();ww=w[m];marker_rows.append({'state':state,'condition':condition,'gene':g,'posterior_weighted_detection':float(np.sum(ww*(v>0))/ww.sum()),'posterior_weighted_mean_count':float(np.sum(ww*v)/ww.sum())})
 tab=pd.DataFrame(rows)
 for state in PROGRAMMES:
  m=tab.state.eq(state);tab.loc[m,'abundance_z']=z(tab.loc[m,'logit_abundance']);tab.loc[m,'programme_z']=z(tab.loc[m,'programme_score'])
 tab['state_contrast']=tab.abundance_z-tab.programme_z;tab.to_csv(out/'donor_state_metrics.csv',index=False);pd.DataFrame(gene_rows).to_csv(out/'target_programme_gene_donor_values.csv',index=False);pd.DataFrame(marker_rows).to_csv(out/'heldout_marker_expression.csv',index=False);pd.DataFrame(uncertainty).to_csv(out/'target_posterior_uncertainty.csv',index=False);return tab,donors,bank
def contrasts(tab,donors,out):
 donors=donors.set_index('donor_id');g=donors.condition.eq('nec').to_numpy();cov=donors[['gestational_age_weeks','age_at_surgery_days']].to_numpy(float);rows=[];un=[];marg=[];permrows=[];wide=tab.pivot(index='donor_id',columns='state',values='state_contrast').loc[donors.index];y=wide[['M cell','LEC']].mean(1).to_numpy()
 for state,v in [('M cell + LEC',y)]+[(s,wide[s].to_numpy()) for s in PROGRAMMES]:
  r,tp=adjusted(v,g,cov);r.update(state=state,family='single primary' if state=='M cell + LEC' else 'three state contrasts',outcome='abundance-programme standardized contrast');rows.append(r);er=exact(v,g);er.update(state=state,model='unadjusted',outcome='state_contrast');un.append(er);permrows.extend({'state':state,'permutation':i+1,'null_studentized_statistic':float(t)} for i,t in enumerate(tp))
 frame=pd.DataFrame(rows);frame['fdr_bh']=frame.p_value;frame.loc[frame.state.ne('M cell + LEC'),'fdr_bh']=bh(frame.loc[frame.state.ne('M cell + LEC'),'p_value']);frame.to_csv(out/'formal_contrast_adjusted.csv',index=False);pd.DataFrame(un).to_csv(out/'formal_contrast_unadjusted.csv',index=False);pd.DataFrame(permrows).to_csv(out/'formal_contrast_permutation_null.csv.gz',index=False,compression='gzip')
 for state in PROGRAMMES:
  sub=tab.loc[tab.state.eq(state)].set_index('donor_id').loc[donors.index]
  for feature in ['logit_abundance','programme_score']:
   rr,_=adjusted(z(sub[feature]),g,cov);rr.update(state=state,outcome=feature);marg.append(rr)
 mf=pd.DataFrame(marg);mf['fdr_bh']=bh(mf.p_value);mf.to_csv(out/'abundance_programme_marginal_effects.csv',index=False);coupling=[]
 for state in PROGRAMMES:
  sub=tab.loc[tab.state.eq(state)];x=stats.rankdata(sub.logit_abundance);y=stats.rankdata(sub.programme_score);r=float(np.corrcoef(x,y)[0,1]);rng=np.random.default_rng(SEED);null=np.array([np.corrcoef(x,rng.permutation(y))[0,1] for _ in range(N_PERM)]);coupling.append({'state':state,'spearman_rho':r,'p_value':float((1+np.sum(np.abs(null)>=abs(r)-1e-12))/(N_PERM+1)),'n':len(sub),'permutations':N_PERM})
 cf=pd.DataFrame(coupling);cf['fdr_bh']=bh(cf.p_value);cf.to_csv(out/'within_donor_coupling.csv',index=False);diagnostics=[]
 for variable in ['gestational_age_weeks','age_at_surgery_days']:
  a=donors.loc[donors.condition.eq('control'),variable];b=donors.loc[donors.condition.eq('nec'),variable];lo=max(a.min(),b.min());hi=min(a.max(),b.max());diagnostics.append({'variable':variable,'control_min':a.min(),'control_max':a.max(),'nec_min':b.min(),'nec_max':b.max(),'interval_lower':lo,'interval_upper':hi,'controls_in_interval':int(a.between(lo,hi).sum()),'nec_in_interval':int(b.between(lo,hi).sum())})
 pd.DataFrame(diagnostics).to_csv(out/'covariate_common_support.csv',index=False);return frame
def validate_reference(root,out):
 rows=[];conf=[];folds=[]
 for state,lineage in LINEAGES.items():
  a=ad.read_h5ad(root/'legacy_analysis/data/processed/dr_cmap'/f'{lineage}_drcmap_input.h5ad');ref=a[a.obs.query_kind.eq('reference')].copy();labels=ref.obs.reference_label.astype(str).to_numpy();donors=ref.obs.donor_id.astype(str).to_numpy();classes=np.array(sorted(set(labels)));features=np.array([g not in HOLDOUT_MARKERS for g in ref.var_names]);raw=ref.X[:,features].tocsr();lib=np.asarray(raw.sum(1)).ravel();norm=raw.multiply((1e4/np.maximum(lib,1))[:,None]).tocsr();norm.data=np.log1p(norm.data);pred=np.empty(len(labels),object)
  for donor in sorted(set(donors)):
   test=donors==donor;train=~test;svd=TruncatedSVD(n_components=min(30,train.sum()-2,norm.shape[1]-2),random_state=SEED);trainx=svd.fit_transform(norm[train]);testx=svd.transform(norm[test]);knn=KNeighborsClassifier(n_neighbors=min(31,int(train.sum())),weights='distance',n_jobs=4);knn.fit(trainx,labels[train]);pred[test]=knn.predict(testx);tp=int(np.sum((labels[test]==state)&(pred[test]==state)));fp=int(np.sum((labels[test]!=state)&(pred[test]==state)));fn=int(np.sum((labels[test]==state)&(pred[test]!=state)));folds.append({'state':state,'heldout_donor':donor,'support':int(np.sum(labels[test]==state)),'train_support':int(np.sum(labels[train]==state)),'true_positive':tp,'false_positive':fp,'false_negative':fn})
  pr,re,f1,sup=precision_recall_fscore_support(labels,pred,labels=classes,zero_division=0)
  for c,p,r,f,s in zip(classes,pr,re,f1,sup):rows.append({'lineage':lineage,'state':c,'is_target':c==state,'support':int(s),'precision':p,'recall':r,'f1':f,'reference_donors':len(set(donors)),'positive_reference_donors':len(set(donors[labels==c])),'excluded_marker_genes':';'.join(sorted(HOLDOUT_MARKERS&set(ref.var_names))),'validation':'donor-held-out train-fitted SVD + kNN; source reference labels'})
  cm=confusion_matrix(labels,pred,labels=classes)
  for i,t in enumerate(classes):
   for j,prediction in enumerate(classes):conf.append({'lineage':lineage,'true_label':t,'predicted_label':prediction,'count':int(cm[i,j])})
  print('validated',lineage,len(labels),'cells',flush=True)
 frame=pd.DataFrame(rows);frame.to_csv(out/'reference_validation_all_classes.csv',index=False);frame.loc[frame.is_target].to_csv(out/'target_validation.csv',index=False);pd.DataFrame(conf).to_csv(out/'reference_validation_confusion.csv',index=False);pd.DataFrame(folds).to_csv(out/'target_validation_donor_folds.csv',index=False)
def lr_specificity(a,donors,out):
 out=out.resolve()
 axes=[('CXCL16-CXCR6','M_cell','CXCL16','T_reg','CXCR6'),('CCL21-CCR7','LEC','CCL21','T_reg','CCR7'),('VEGFC-FLT4','fibroblast','VEGFC','LEC','FLT4'),('EFNA1-EPHA2','LEC','EFNA1','epithelial','EPHA2'),('TNF-ICAM1','myeloid','TNF','epithelial','ICAM1'),('IL22-IL22RA1','T_NK','IL22','epithelial','IL22RA1')];obs=a.obs.copy();obs.donor_id=obs.donor_id.astype(str);counts=a.layers['counts'].tocsr();g=donors.condition.eq('nec').to_numpy();ids=donors.donor_id.to_list();banks={};detections={}
 for state in sorted(set(s for _,s,_,_,_ in axes)|set(t for _,_,_,t,_ in axes)):
  mat=[];detect=[]
  for d in ids:
   m=obs.donor_id.eq(d)&obs.selection.eq('unselected')&obs.cell_state.eq(state)
   if m.sum()<10:
    mat.append(np.full(a.n_vars,np.nan));detect.append(np.full(a.n_vars,np.nan));continue
   mat.append(np.asarray(a.X[m.to_numpy()].mean(0)).ravel());detect.append(np.asarray((counts[m.to_numpy()]>0).sum(0)).ravel()/int(m.sum()))
  banks[state]=np.vstack(mat);detections[state]=np.nanmean(detect,axis=0)
 resource=pd.read_csv(out.parent/'inputs/liana_native_rank_aggregate.csv',usecols=['receptor_complex']);receptors=set(g for v in resource.receptor_complex.astype(str) for g in v.split('_'));receptor_universe=np.array([g in receptors for g in a.var_names]);rng=np.random.default_rng(SEED);effects=[];nullrows=[];scores=[]
 for axis,source,ligand,target,receptor in axes:
  li=a.var_names.get_loc(ligand);ri=a.var_names.get_loc(receptor);lvfull=banks[source][:,li];rvfull=banks[target][:,ri];valid=np.isfinite(lvfull)&np.isfinite(rvfull);lv=lvfull[valid];rv=rvfull[valid];gv=g[valid];score=(z(lv)+z(rv))/2;er=exact(score,gv);observed=abs(er['hedges_g']);mean=np.nanmean(banks[target],axis=0);det=detections[target];eligible=(det>0)&np.isfinite(mean)&receptor_universe;rank=stats.rankdata(mean)/len(mean);distance=np.abs(rank-rank[ri])+np.abs(det-det[ri]);distance[~eligible]=np.inf;distance[ri]=np.inf;candidate=np.argsort(distance)[:min(100,int(eligible.sum())-int(eligible[ri]))];sample=rng.choice(candidate,size=999,replace=True);null=[]
  for k,j in enumerate(sample):
   ns=(z(lv)+z(banks[target][valid,j]))/2;effect=hedges(ns,gv);null.append(effect);nullrows.append({'axis':axis,'draw':k+1,'random_gene':str(a.var_names[j]),'null_hedges_g':float(effect),'control':'expression/detection-matched random receptor from LIANA-tested universe'})
  effects.append({'axis':axis,'source_state':source,'ligand':ligand,'target_state':target,'receptor':receptor,**er,'n_nec':int(gv.sum()),'n_control':int((~gv).sum()),'minimum_cells_per_state_donor':10,'specificity_p':(1+np.sum(np.abs(null)>=observed-1e-12))/1000,'random_sets':999,'candidate_pool':len(candidate),'control_note':'100 nearest detection/expression-matched expressed receptors from LIANA-tested universe','class':'ligand-target' if axis=='TNF-ICAM1' else 'ligand-receptor'})
  for d,c,s,l,r in zip(np.asarray(ids)[valid],donors.condition.to_numpy()[valid],score,lv,rv):scores.append({'donor_id':d,'condition':c,'axis':axis,'score':s,'ligand_mean_log_expression':l,'receptor_mean_log_expression':r})
 f=pd.DataFrame(effects);f['fdr_bh']=bh(f.p_value);f['specificity_fdr_bh']=bh(f.specificity_p);f.to_csv(out/'LR_specificity.csv',index=False);pd.DataFrame(nullrows).to_csv(out/'LR_specificity_null.csv',index=False);pd.DataFrame(scores).to_csv(out/'LR_donor_scores.csv',index=False);ccl=[]
 for state in ['LEC','vascular_endothelial']:
  for d,cond in zip(ids,donors.condition):
   m=obs.donor_id.eq(d)&obs.selection.eq('unselected')&obs.cell_state.eq(state);cnt=np.asarray(counts[m.to_numpy()].sum(0)).ravel();ix=a.var_names.get_loc('CCL21');value=float(np.log2((cnt[ix]+.5)/(cnt.sum()+1)*1e6)) if m.sum()>=10 else np.nan;ccl.append({'donor_id':d,'condition':cond,'state':state,'n_cells':int(m.sum()),'CCL21_log2_cpm':value,'eligible':bool(m.sum()>=10)})
 pd.DataFrame(ccl).to_csv(out/'CCL21_vascular_donor_values.csv',index=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--posteriors',type=Path);p.add_argument('--skip-validation',action='store_true');args=p.parse_args();root=args.root;out=root/'new_results';out.mkdir(exist_ok=True);start=time.time();posterior=args.posteriors or root/'inputs/native_posteriors.csv.gz';atlas=root/'04_corrected_results/01_single_cell_corrected/corrected_hca_nec_atlas.h5ad';a=ad.read_h5ad(atlas);post=pd.read_csv(posterior,dtype={'donor_id':str});meta=pd.read_csv(root/'inputs/donor_metadata.csv',dtype={'donor_id':str});meta['postconception_age_weeks']=meta.gestational_age_weeks+meta.age_at_surgery_days/7;meta['segment']='small intestine (source study)';meta['Bell_stage']='not reported';meta['time_onset_to_resection']='not reported';meta['antibiotics_at_sampling']='not reported';meta['metadata_source']='Egozi et al. 2023 PLOS Biology, Supplementary Table S1';meta.to_csv(out/'donor_covariates.csv',index=False);tab,donors,bank=weighted_tables(a,post,meta,out);f=contrasts(tab,donors,out);print(f.to_string(index=False),flush=True);lr_specificity(a,donors,out)
 if not args.skip_validation:validate_reference(root,out)
 manifest={'completed':True,'seconds':time.time()-start,'platform':platform.platform(),'posterior_source':str(posterior),'counts_source':str(atlas),'seed':SEED,'permutations':N_PERM,'primary':f.iloc[0].to_dict(),'upstream_note':'QC atlas integer-count checkpoint; full 10x rerun logged separately; saved scANVI inference provenance recorded separately'};(out/'revision_analysis_summary.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('DONE',manifest['seconds'],flush=True)
if __name__=='__main__':main()
