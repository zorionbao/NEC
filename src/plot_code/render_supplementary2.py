"""Render developmental context and donor-model diagnostics from cloud results."""
import argparse,json
from pathlib import Path
from render_revision_figures import *
p=argparse.ArgumentParser();p.add_argument('--new-results',type=Path,required=True);p.add_argument('--story-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.new_results=args.new_results.resolve();args.story_root=args.story_root.resolve();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
files={k:str(args.new_results/f) for k,f in {'donors':'primary_contrast_donor_values.csv','models':'primary_contrast_model_comparison.csv','lodo':'primary_contrast_leave_one_donor.csv'}.items()}
files.update({k:str(args.story_root/'specific_nec'/f) for k,f in {'matched_quantiles':'specific_nec_matched_null_quantiles.csv','matched_summary':'specific_nec_matched_null_module_summary.csv','bulk_modules':'specific_nec_bulk_disease_control_module_summary.csv'}.items()})
mp=args.out/'supplementary2_inputs.json';mp.write_text(json.dumps({'inputs':files,'cloud_execution':'cloud statistical outputs'},indent=2),encoding='utf-8');inp=Inputs(mp,args.out);fig=canvas(250);glegend(fig)
donors=inp.get('donors')
for letter,xlabel,xc,box in [('A','Gestational age (weeks)','gestational_age_weeks',[.14,.745,.32,.18]),('B','Age at surgery (days)','age_at_surgery_days',[.64,.745,.30,.18])]:
 a=panel(fig,box,letter,'Combined donor contrast')
 for group,d in donors.groupby('condition'):a.scatter(d[xc],d.primary_contrast,s=25,color=GROUP[group],edgecolor='white',lw=.4)
 a.set_xlabel(xlabel);a.set_ylabel('Donor contrast');clean(a);inp.save('S2',letter,donors)
a=panel(fig,[.31,.40,.15,.18],'C','Diagnosis-effect estimates');d=inp.get('models');d['display']=d.model.str.replace('GA and surgery age adjusted','Age adjusted',regex=False);forest(a,d,ef='coefficient',lo='ci_low',hi='ci_high',label='display',xlabel='Diagnosis coefficient');inp.save('S2','C',d)
a=panel(fig,[.70,.40,.24,.18],'D','Donor influence on\nadjusted estimate');d=inp.get('lodo');d['display']=d.omitted_donor.astype(str).str.replace('D','',regex=False);forest(a,d,ef='coefficient',lo='_none',hi='_none',label='display',xlabel='Coefficient after donor omission');a.set_ylabel('Omitted donor');inp.save('S2','D',d)
a=panel(fig,[.14,.10,.32,.18],'E','Selective transcription programs');d=inp.get('matched_quantiles');summary=inp.get('matched_summary');colors=[OI[x] for x in ['blue','sky','green','orange','red']]
for (module,g),color in zip(d.groupby('module',sort=False),colors):
 a.plot(g.aligned_mean_log2_fold_change,g['quantile'],color=color,lw=1.2,label=module)
 r=summary.loc[summary.module.eq(module)].iloc[0];a.scatter(r.observed_aligned_mean_log2_fold_change,r.matched_null_percentile,color=color,marker='D',s=20,zorder=3)
a.set_xlabel('Expression-matched null effect');a.set_ylabel('Cumulative probability');clean(a);a.legend(loc='upper center',bbox_to_anchor=(.5,-.27),ncol=2,columnspacing=.6,handlelength=.8,handletextpad=.3);inp.save('S2','E',d);inp.save('S2','E_summary',summary)
a=panel(fig,[.68,.10,.20,.18],'F','Bowel-tissue program effects');d=inp.get('bulk_modules');d['display']=[str(a)+'\nNEC/'+('SIP' if 'perforation' in str(c).lower() or 'sip' in str(c).lower() else 'control') for a,c in zip(d.dataset,d.contrast)];mat=d.pivot_table(index='display',columns='module',values='mean_aligned_log2_fold_change');mat=mat.reindex(index=['GSE46619'+chr(10)+'NEC/control','GSE46619'+chr(10)+'NEC/SIP','GSE64801'+chr(10)+'NEC/control'],columns=['Antigen sampling','Lymphatic support','Immune residency','Epithelial repair','Inflammatory injury']);mat.columns=[x.replace(' ','\n') for x in mat.columns];heat(fig,a,mat,-4,4,'Aligned mean log2 fold change',90);inp.save('S2','F',d);finish(fig,'Supplementary Figure 2',inp)
print('Supplementary Figure 2 rendered',flush=True)
