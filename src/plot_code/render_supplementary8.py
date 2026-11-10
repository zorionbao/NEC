"""Rebuild annotation-uncertainty panels from complete native scANVI outputs.
The panel/statistic definitions follow plot_extended_data7 in the existing 46 script.
"""
import argparse,json
from pathlib import Path
from render_revision_figures import *
p=argparse.ArgumentParser();p.add_argument('--uplift-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.uplift_root=args.uplift_root.resolve();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
files={k:str(args.uplift_root/f) for k,f in {'posterior':'uplift_full_scanvi_posteriors.csv.gz','donors':'uplift_donor_abundance.csv','abundance_effects':'uplift_abundance_effects.csv','draws':'uplift_abundance_imputation_draws.csv.gz','genes':'uplift_target_gene_effects.csv'}.items()};mp=args.out/'supplementary8_inputs.json';mp.write_text(json.dumps({'inputs':files,'cloud_execution':'native posterior propagation'},indent=2),encoding='utf-8');inp=Inputs(mp,args.out)
post=inp.get('posterior');donor=inp.get('donors');draws=inp.get('draws');genes=inp.get('genes');colors={'epithelial':OI['orange'],'endothelial':OI['green'],'immune':OI['blue']};targets=['M_cell','LEC','T_reg'];fig=canvas(250)
handles=[Line2D([],[],color=colors[l],lw=1.5,label=t) for l,t in [('epithelial','Epithelial / M cells'),('endothelial','Endothelial / LECs'),('immune','Immune / Tregs')]];fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.992),ncol=3,handlelength=1.2,columnspacing=.8,handletextpad=.3)
def kde(values):
 v=np.asarray(values,float);v=v[np.isfinite(v)];sd=v.std(ddof=1)
 if len(v)<2 or sd==0:raise ValueError('Degenerate posterior density')
 pad=.18*(v.max()-v.min());x=np.linspace(v.min()-pad,v.max()+pad,250);bw=sd*len(v)**(-.2);y=np.exp(-.5*((x[:,None]-v[None,:])/bw)**2).mean(1)/(bw*np.sqrt(2*np.pi));return x,y
def g_effect(x,c):
 a=x[c];b=x[~c];df=len(a)+len(b)-2;sd=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/df);return (a.mean()-b.mean())/sd*(1-3/(4*df-1))
a=panel(fig,[.14,.765,.32,.17],'A','Posterior concentration')
for lineage,color in colors.items():
 v=post.loc[post.lineage_model.eq(lineage),'posterior_max_probability'];x,y=kde(v);a.plot(x,y,color=color,lw=1.2);a.fill_between(x,y,color=color,alpha=.12)
a.set_xlim(0,1.01);a.set_xlabel('Maximum class probability');a.set_ylabel('Density');clean(a);inp.save('S8','A',post.drop(columns=['previous_scanvi_label'],errors='ignore'))
a=panel(fig,[.65,.765,.30,.17],'B','Cell-level posterior uncertainty')
for lineage,color in colors.items():
 d=post.loc[post.lineage_model.eq(lineage)];a.scatter(d.posterior_max_probability,d.posterior_entropy_normalized,s=1.2,color=color,alpha=.15,lw=0,rasterized=True)
a.set_xlim(0,1.01);a.set_ylim(0,1.01);a.set_xlabel('Maximum class probability');a.set_ylabel('Normalized entropy');clean(a);inp.save('S8','B',post[['cell_id_original','donor_id','lineage_model','posterior_max_probability','posterior_entropy_normalized']])
a=panel(fig,[.14,.46,.32,.17],'C','Donor target-state abundance')
for state in targets:
 d=donor[donor.target_state.eq(state)];a.scatter(d.hard_fraction_all_cells*100,d.posterior_fraction_all_cells*100,s=24,color=TCOLOR[state],edgecolor='white',lw=.35)
limit=max(donor.hard_fraction_all_cells.max(),donor.posterior_fraction_all_cells.max())*106;a.plot([0,limit],[0,limit],color='black',ls=':',lw=.6);a.set_xlim(0,limit);a.set_ylim(0,limit);a.set_xlabel('Hard-label abundance (%)');a.set_ylabel('Posterior abundance (%)');clean(a);inp.save('S8','C',donor)
effects=[]
for (state,assignment),d in draws.groupby(['target_state','imputation'],sort=False):
 assert len(d)==11;v=np.log((d.imputed_count.to_numpy()+.5)/(d.total_unselected_cells.to_numpy()-d.imputed_count.to_numpy()+.5));effects.append({'target_state':state,'assignment':assignment,'hedges_g':g_effect(v,d.condition.eq('nec').to_numpy())})
effect=pd.DataFrame(effects);assert len(effect)==750
a=panel(fig,[.65,.46,.30,.17],'D','Assignment effect distributions')
for state in targets:
 x,y=kde(effect.loc[effect.target_state.eq(state),'hedges_g']);a.plot(x,y,color=TCOLOR[state],lw=1.2);a.fill_between(x,y,color=TCOLOR[state],alpha=.12)
a.set_xlabel('Abundance Hedges g');a.set_ylabel('Density');clean(a);inp.save('S8','D',effect);inp.save('S8','D_assignments',draws)
hard=genes.loc[genes.method.eq('Hard scANVI label')].set_index(['target_state','gene']);weighted=genes.loc[genes.method.eq('Posterior expectation')].set_index(['target_state','gene']);comparison=hard[['hedges_g_nec_vs_control']].join(weighted[['hedges_g_nec_vs_control']],lsuffix='_hard',rsuffix='_weighted').reset_index();assert len(comparison)==23
a=panel(fig,[.12,.14,.18,.17],'E','Hard and weighted'+chr(10)+'gene effects')
for state in targets:
 d=comparison[comparison.target_state.eq(state)];a.scatter(d.hedges_g_nec_vs_control_hard,d.hedges_g_nec_vs_control_weighted,s=25,color=TCOLOR[state],edgecolor='white',lw=.35)
v=comparison[['hedges_g_nec_vs_control_hard','hedges_g_nec_vs_control_weighted']].to_numpy();bounds=[np.nanmin(v)-.2,np.nanmax(v)+.2];a.plot(bounds,bounds,color='black',ls=':',lw=.6);a.set_xlim(bounds);a.set_ylim(bounds);a.set_xlabel('Hard-label gene Hedges g');a.set_ylabel('Weighted gene Hedges g');clean(a);inp.save('S8','E',comparison)
a=panel(fig,[.51,.14,.37,.17],'F','Gene effects by annotation strategy');methods=['Hard scANVI label','Posterior expectation','Posterior multiple imputation'];gene_order=genes.loc[genes.method.eq(methods[2]),'gene'].tolist();genes['display_effect']=np.where(genes.method.eq(methods[2]),genes.mi_effect_median,genes.hedges_g_nec_vs_control);m=genes.pivot_table(index='method',columns='gene',values='display_effect').reindex(index=methods,columns=gene_order);assert m.notna().all().all();m.index=['Hard labels','Posterior weights','Imputation median'];heat(fig,a,m,-2,2,'Hedges g',90);inp.save('S8','F',genes);finish(fig,'Supplementary Figure 8',inp)
print('Supplementary Figure 8 rendered: complete native posteriors, 11 donors, 250 assignments',flush=True)
