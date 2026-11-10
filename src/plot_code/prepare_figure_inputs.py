"""Prepare plot tables from verified cloud computation outputs."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--stats-root',type=Path,required=True);p.add_argument('--single-cell',type=Path,required=True);p.add_argument('--new-results',type=Path,required=True);p.add_argument('--story-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--figures',nargs='+',type=int,default=[1,2,3,4]);args=p.parse_args()
S=args.stats_root;N=args.new_results;T=args.story_root;O=args.out;O.mkdir(parents=True,exist_ok=True);inputs={}
def add(k,p):
 if not p.is_file():raise FileNotFoundError(p)
 inputs[k]=str(p.resolve())
def save(k,d):
 f=O/(k+'.csv');d.to_csv(f,index=False);add(k,f)
if 1 in args.figures:
 for k,f in {'atlas':'corrected_single_cell_metadata_scores_umap.csv.gz'}.items():add(k,args.single_cell/f)
for k,f in {'composition':'single_cell_donor_composition.csv','axis_effects':'sample_level_mechanistic_axis_effects.csv','axis_scores':'sample_level_mechanistic_axis_scores.csv','imaging_fields':'tissue_tunel_lyve1_hpf_source.csv','imaging_subjects':'tissue_tunel_lyve1_subject_level.csv','species_effects':'metaphlan_species_effects_corrected.csv','species_long':'metaphlan_species_sample_long_corrected.csv','pathway_effects':'humann_pathway_selected_panel.csv','pathway_long':'humann_pathway_selected_sample_long.csv','metabolite_effects':'target_metabolite_patient_effects.csv','metabolite_correlations':'wandro_patient_level_taxon_metabolite_correlations.csv'}.items():add(k,S/f)
for k,f in {'donor_covariates':'donor_covariates.csv','target_validation':'target_validation.csv','donor_state_metrics':'donor_state_metrics.csv','contrast_effects':'formal_contrast_adjusted.csv'}.items():add(k,N/f)
vascular=pd.read_csv(N/'normalized_donor_expression_selected.csv',dtype={'donor_id':str})
vascular=vascular.loc[vascular.state.eq('native_vascular_endothelial')&vascular.gene.isin(['SELE','VCAM1'])].copy()
vascular=vascular.rename(columns={'TMM_log2_CPM':'log2_cpm'})
vascular['eligible']=vascular.n_cells.ge(10)&np.isfinite(vascular.log2_cpm)
save('native_vascular_markers',vascular)
if 4 in args.figures:
 ge=pd.read_csv(N/'programme_gene7_adjusted.csv');support=pd.read_csv(N/'programme_gene_count_support.csv');ge=ge.drop(columns=['nonzero_donors'],errors='ignore').merge(support[['state','gene','donors','zero_count_donors','median_weighted_count','maximum_weighted_count']],on=['state','gene'],how='left');ge['nonzero_donors']=ge.donors-ge.zero_count_donors;assert ge.nonzero_donors.notna().all();save('target_gene_effects',ge)
add('pathway_all_effects',S/'humann_pathway_effects_corrected.csv')
add('metabolite_all_effects',S/'metabolite_patient_level_effects.csv')
assert len(pd.read_csv(S/'humann_pathway_effects_corrected.csv'))==383
assert len(pd.read_csv(S/'metabolite_patient_level_effects.csv'))==453
meta=pd.read_csv(N/'donor_covariates.csv',dtype={'donor_id':str});condition=meta.set_index('donor_id').condition.to_dict()
host=pd.read_csv(S/'single_cell_host_state_effects.csv');modules=pd.read_csv(S/'single_cell_module_effects.csv');composition=pd.read_csv(S/'single_cell_composition_effects.csv');composition=composition.loc[composition.family.eq('single_cell_composition_clr_abundance')&composition.feature.isin(['M_cell','LEC','T_reg','T_NK'])].copy()
host['display_label']=host.feature.map({'surveillance_state':'Surveillance state','injury_state':'Inflammatory injury state','injury_surveillance_imbalance':'Injury-surveillance imbalance'});modules['display_label']=modules.feature;composition['display_label']=composition.feature.map({'M_cell':'M cells CLR abundance','LEC':'Lymphatic endothelial CLR abundance','T_reg':'Regulatory T cells CLR abundance','T_NK':'T and NK cells CLR abundance'})
d=pd.concat([host,modules,composition],ignore_index=True);assert len(d)==12,len(d)
v=d.p_value.to_numpy();ix=np.argsort(v);q=np.minimum.accumulate((v[ix]*len(v)/np.arange(1,len(v)+1))[::-1])[::-1];bh=np.empty(len(v));bh[ix]=np.minimum(q,1);d['fdr_bh']=bh;save('host_effects',d.sort_values('hedges_g'))
if (N/'display_allcell_gene_donor_logcpm.csv').is_file():
 d=pd.read_csv(N/'display_allcell_gene_donor_logcpm.csv',dtype={'donor_id':str});genes=[c for c in d if c not in ['donor_id','condition','captured_cells','library_size']];d[genes]=(d[genes]-d[genes].mean(0))/d[genes].std(0,ddof=0);save('host_gene_zscores',d)
elif 1 in args.figures:raise FileNotFoundError(N/'display_allcell_gene_donor_logcpm.csv')
else:
 d=pd.read_csv(S/'single_cell_pseudobulk_target_logcpm_all_cells.csv').set_index('gene').T;d.index.name='donor_id';d=(d-d.mean(0))/d.std(0,ddof=0);d=d.reset_index();d['donor_id']=d.donor_id.astype(str).str.extract(r'(\d+)',expand=False);d['condition']=d.donor_id.map(condition);assert len(d)==11 and d.condition.notna().all();save('host_gene_zscores',d)
frames=[]
for st in ['M_cell','LEC','T_reg','T_NK','epithelial','myeloid','vascular_endothelial']:
 f=N/f'adjusted_edger_{st}.csv'
 if f.is_file():frames.append(pd.read_csv(f).assign(cell_state=st))
save('state_de',pd.concat(frames,ignore_index=True))
bulk=T/'state_resolve/state_resolve_target_gene_effects.csv';bulk_frame=pd.read_csv(bulk);bulk_frame=bulk_frame.loc[bulk_frame.model.astype(str).str.contains('composition',case=False)].copy();assert len(bulk_frame)>0;save('bulk_de',bulk_frame)
d=pd.read_csv(N/'LR_specificity_null.csv');d=d.loc[d.axis.eq('CCL21-CCR7')].rename(columns={'null_hedges_g':'effect'}).assign(kind='null');obs=pd.read_csv(N/'LR_specificity.csv');obs=obs.loc[obs.axis.eq('CCL21-CCR7')].rename(columns={'hedges_g':'effect'}).assign(kind='observed');save('lr_specificity',pd.concat([obs,d],ignore_index=True))
b=bulk_frame.copy();b=b.loc[b.accession.eq('GSE46619')&b.contrast.astype(str).str.contains('control',case=False)].copy();b=b[['gene','logFC']].rename(columns={'logFC':'bulk_log2fc'});s=pd.read_csv(S/'single_cell_pseudobulk_target_genes_all_cells.csv')[['gene','logFC']].rename(columns={'logFC':'single_cell_log2fc'});save('concordance',s.merge(b,on='gene'))
if 4 in args.figures:
 for k,f in {'module_effects':'bridge_nec/bridge_primary_module_source_effects.csv','donor_stage':'gate_ot/gate_ot_donor_maturation_stage.csv','stage_probability':'gate_ot/gate_ot_donor_stage_probabilities.csv'}.items():add(k,T/f)
(O/'figure_inputs.json').write_text(json.dumps({'inputs':inputs,'cloud_execution':{'status':'cloud recomputation outputs','new_results':str(N),'stats_root':str(S),'story_root':str(T)}},indent=2),encoding='utf-8')
print('Prepared',len(inputs),'figure inputs',flush=True)
