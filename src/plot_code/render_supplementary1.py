"""Six atlas diagnostics with current UMAP and full-fit classifier confidence."""
import argparse, hashlib, json
from pathlib import Path
from render_revision_figures import *

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--reviewer-tables',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 font_manager.findfont('Liberation Sans',fallback_to_default=False)
 paths={'atlas':args.raw_root/'corrected_single_cell_metadata_scores_umap.csv.gz',**{k:args.reviewer_tables/f for k,f in {'markers':'marker_dotplot_data.csv','capture':'capture_mode_paired_values.csv','sensitivity':'single_cell_covariate_sensitivity.csv'}.items()}}
 mp=args.out/'supplementary1_inputs.json';mp.write_text(json.dumps({'inputs':{k:str(v.resolve()) for k,v in paths.items()}},indent=2),encoding='utf-8');inp=Inputs(mp,args.out)
 meta=inp.get('atlas');markers=inp.get('markers');capture=inp.get('capture');sensitivity=inp.get('sensitivity')
 markers['cell_state']=markers.cell_state.map(LABEL).fillna(markers.cell_state)
 sp=args.raw_root/'single_cell_rebuild_status.json';status=json.loads(sp.read_text(encoding='utf-8'));transfer=status['label_transfer']['leave_one_donor_out'];labels=transfer['confusion_labels'];counts=np.asarray(transfer['confusion_matrix'],float)
 confusion=pd.DataFrame(counts/np.maximum(counts.sum(axis=1,keepdims=True),1),index=labels,columns=labels).reindex(index=STATES,columns=STATES).fillna(0)
 qc=meta.groupby(['run_id','donor_id','condition','selection'],as_index=False).agg(cells_after_qc=('run_id','size'),median_total_counts=('total_counts','median'),median_genes_detected=('n_genes_by_counts','median'))
 fig=canvas(250)
 a=panel(fig,[.25,.855,.66,.11],'A','Multi-marker expression across cell states')
 genes=markers[['marker_family','gene']].drop_duplicates().sort_values(['marker_family','gene']).gene.tolist();states=[LABEL[s] for s in STATES]
 a.scatter(markers.gene.map({g:i for i,g in enumerate(genes)}),markers.cell_state.map({s:i for i,s in enumerate(states)}),s=3+markers.fraction_expressing.clip(lower=0)*29,c=markers.mean_expression_z,cmap=CMAP,vmin=-2,vmax=2,edgecolors='black',linewidths=.15)
 a.set_xticks(range(len(genes)),genes,rotation=90);a.set_yticks(range(len(states)),states);a.set_xlim(-.6,len(genes)-.4);a.set_ylim(len(states)-.5,-.5);a.tick_params(length=0,pad=2)
 for spine in a.spines.values():spine.set_visible(False)
 ca=fig.add_axes([.925,.855,.012,.11]);cb=fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-2,2),cmap=CMAP),cax=ca);cb.set_ticks([-2,0,2]);cb.set_label('Expression z score',fontsize=8);cb.ax.tick_params(labelsize=8)
 handles=[Line2D([],[],marker='o',ls='',markerfacecolor='white',markeredgecolor='black',markeredgewidth=.4,markersize=np.sqrt(3+v*29),label=f'{v:.0%}') for v in [.25,.5,.75]]
 fig.legend(handles=handles,title='Cells expressing',loc='upper right',bbox_to_anchor=(.89,.997),ncol=3,columnspacing=.7,handletextpad=.3,title_fontsize=8);inp.save('S1','A',markers)
 a=panel(fig,[.14,.60,.31,.125],'B','Run-level cell quality')
 for (condition,selection),t in qc.groupby(['condition','selection']):
  a.scatter(t.median_total_counts,t.median_genes_detected,s=18+np.sqrt(t.cells_after_qc)*.26,color=GROUP[condition],marker='o' if selection=='unselected' else 's',edgecolors='black',linewidths=.3)
 a.set_xlabel('Median UMI count');a.set_ylabel('Median detected genes');clean(a)
 a.legend(handles=[Line2D([],[],marker='o',ls='',color=GROUP[g],label=l,ms=4) for g,l in [('control','Control'),('nec','NEC')]],loc='upper left',handletextpad=.3)
 fig.legend(handles=[Line2D([],[],marker=m,ls='',markerfacecolor='white',markeredgecolor='black',label=l,ms=4) for m,l in [('o','Unselected'),('s','CD45 depleted')]],loc='upper left',bbox_to_anchor=(.125,.575),ncol=2,columnspacing=.6,handletextpad=.3);inp.save('S1','B',qc)
 a=panel(fig,[.64,.60,.31,.125],'C','Label-transfer confidence')
 a.scatter(meta.umap_1,meta.umap_2,c=meta.label_transfer_confidence,s=.35,cmap=SEQ,vmin=0,vmax=1,linewidths=0,rasterized=True)
 a.set_xlabel('UMAP 1');a.set_ylabel('UMAP 2');a.set_xticks([]);a.set_yticks([]);clean(a)
 ca=fig.add_axes([.72,.57,.18,.006]);cb=fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0,1),cmap=SEQ),cax=ca,orientation='horizontal');cb.set_ticks([0,1]);cb.ax.tick_params(labelsize=8,length=2,pad=1)
 inp.save('S1','C',meta[['cell','donor_id','condition','cell_state','label_transfer_confidence','umap_1','umap_2']])

 a=panel(fig,[.215,.34,.28,.18],'D','Held-out state concordance')
 im=a.imshow(confusion.to_numpy(),cmap=SEQ,vmin=0,vmax=1,aspect='auto',interpolation='nearest');compact=['M cells','LECs','Tregs','T/NK','Epithelial','Epithelial stem','Myeloid','Vascular EC','Fibroblast']
 a.set_xticks(range(9),compact,rotation=90);a.set_yticks(range(9),compact);a.tick_params(length=0,pad=2)
 for spine in a.spines.values():spine.set_visible(False)
 ca=fig.add_axes([.51,.34,.012,.18]);cb=fig.colorbar(im,cax=ca);cb.set_ticks([0,.5,1]);cb.set_label('Row fraction',fontsize=8);cb.ax.tick_params(labelsize=8);inp.save('S1','D',confusion.rename_axis('true_state').reset_index())
 paired=capture.copy();paired['feature_mean']=paired.groupby('feature').value.transform('mean');paired['feature_sd']=paired.groupby('feature').value.transform('std').replace(0,1);paired['z_value']=(paired.value-paired.feature_mean)/paired.feature_sd
 wide=paired.pivot_table(index=['feature_family','feature','donor_id'],columns='capture_mode',values='z_value').reset_index();wide['difference']=wide.CD45_depleted-wide.unselected
 features=wide.groupby('feature').difference.median().sort_values().index.tolist();short={'Lymphatic endothelial cells':'Lymphatic endothelial'}
 a=panel(fig,[.79,.34,.16,.18],'E','Paired capture effects');colors={'Cell-state fraction':OI['blue'],'Gene-program score':OI['red']}
 for i,feature in enumerate(features):
  t=wide[wide.feature.eq(feature)];color=colors[t.feature_family.iloc[0]];a.scatter(t.difference,np.full(len(t),i),s=11,c=color,edgecolors='black',linewidths=.2);a.scatter([t.difference.median()],[i],s=30,marker='D',facecolors='white',edgecolors=color,linewidths=1,zorder=3)
 a.set_yticks(range(len(features)),[short.get(x,x) for x in features]);a.invert_yaxis();a.set_xlabel('Paired z-score difference');a.set_xticks([-3,0,3]);clean(a)
 fig.legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=l,ms=4) for l,c in colors.items()]+[Line2D([],[],marker='D',ls='',markerfacecolor='white',markeredgecolor='black',label='Median',ms=4)],title='Paired-capture features (E)',loc='upper center',bbox_to_anchor=(.56,.795),ncol=3,handletextpad=.4,columnspacing=.8,title_fontsize=8);inp.save('S1','E',wide)

 a=panel(fig,[.43,.045,.42,.18],'F','Donor-level covariate sensitivity');models=['Unadjusted','Gestational age adjusted','Age at surgery adjusted'];features=sensitivity.feature.drop_duplicates().tolist();s=sensitivity.copy();s['minus_log10_p']=-np.log10(s.p_value.clip(lower=1e-300));pmax=max(float(s.minus_log10_p.max()),1);limit=max(1.5,float(s.standardized_nec_beta.abs().max()))
 a.scatter(s.adjustment.map({x:i for i,x in enumerate(models)}),s.feature.map({x:i for i,x in enumerate(features)}),c=s.standardized_nec_beta,s=6+36*s.minus_log10_p/pmax,cmap=CMAP,vmin=-limit,vmax=limit,edgecolors='black',linewidths=.2)
 a.set_xticks(range(3),['Unadjusted','Gestational age','Surgery age']);a.set_yticks(range(len(features)),features);a.set_xlim(-.5,2.5);a.set_ylim(len(features)-.5,-.5);a.tick_params(length=0,pad=3)
 for spine in a.spines.values():spine.set_visible(False)
 ca=fig.add_axes([.88,.045,.012,.18]);cb=fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-limit,limit),cmap=CMAP),cax=ca);cb.set_label('Standardized NEC coefficient',fontsize=8);cb.ax.tick_params(labelsize=8)
 fig.text(.66,.248,'Dot area: -log10 P',fontsize=8,ha='center');inp.save('S1','F',s)
 finish(fig,'Supplementary Figure 1',inp)
 inp.sources.append({'key':'held_out_confusion','path':str(sp.resolve()),'sha256':hashlib.sha256(sp.read_bytes()).hexdigest()})
 (args.out/'figure_input_manifest.json').write_text(json.dumps({'sources':inp.sources,'cells':len(meta),'runs':len(qc),'confidence_definition':'Maximum probability from classifier fitted to all known reference cells; separate from held-out confusion.','status':'rendered from current raw rebuild'},indent=2),encoding='utf-8')
 print('Supplementary Figure 1 rendered',flush=True)

if __name__=='__main__':main()
