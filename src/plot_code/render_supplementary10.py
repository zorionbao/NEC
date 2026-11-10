"""Render final ECLIPSE tables; numerical statistics are never modified here."""
from pathlib import Path
import importlib.util, json, textwrap, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('original_eclipse_plot',ROOT/'legacy_analysis/code/27_plot_eclipse_nec.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for p in (Path(__file__).parent/'fonts').glob('*.ttf'):
    font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'axes.grid':False,'savefig.bbox':None})
out=ROOT/'publication_figures/rendered'; out.mkdir(parents=True,exist_ok=True)
source=ROOT/'publication_figures/source_data/Supplementary_Figure_10'; source.mkdir(parents=True,exist_ok=True)
res=m.RESULT_ROOT
meta_raw=pd.read_parquet(res/'eclipse_cross_tissue_window_meta.parquet')
meta=m.manhattan_coordinates(meta_raw)
tissue=m.top_window_tissue_matrix(meta_raw)
signals=pd.read_csv(res/'eclipse_cross_tissue_promoter_signal_meta.csv')
modules=pd.read_csv(res/'eclipse_cell_state_module_summary.csv')
pca=pd.read_csv(res/'eclipse_stool_methylation_pca.csv')
scores=pd.read_csv(res/'eclipse_stool_nec_axis_projection.csv')
for name,frame in [('A_cross_tissue_windows',meta_raw),('B_top_window_tissue_effects',tissue),('C_promoter_signals',signals),('D_module_correlations',modules),('E_residual_pca',pca),('F_nested_projection',scores)]:
    if name.startswith('A'): frame.to_parquet(source/(name+'.parquet'),index=False)
    else: frame.to_csv(source/(name+'.csv'),index=False)

fig=plt.figure(figsize=(180/25.4,275/25.4))
positions=[[.12,.745,.43,.19],[.77,.745,.13,.19],[.12,.425,.43,.19],[.77,.425,.13,.19],[.12,.105,.43,.19],[.70,.105,.25,.19]]
ax=[fig.add_axes(pos) for pos in positions]
c=m.OKABE_ITO
for i,chrom in enumerate(m.chromosome_order(meta.chromosome)):
    d=meta[meta.chromosome.eq(chrom)]
    ax[0].scatter(d.genome_position,d.minus_log10_p,s=.45,c=c['blue'] if i%2==0 else c['sky_blue'],alpha=.55,linewidths=0,rasterized=True)
sig=meta[meta.meta_fdr.lt(.05)]
ax[0].scatter(sig.genome_position,sig.minus_log10_p,s=18,c=c['vermillion'],edgecolors='black',linewidths=.4,zorder=4)
for i,row in enumerate(sig.itertuples()):
    ax[0].annotate(f'{row.chromosome}:{row.bin_start/1e6:.3f} Mb',(row.genome_position,row.minus_log10_p),xytext=(2,8+8*i),textcoords='offset points',fontsize=8)
ticks=meta.attrs['midpoints']; ticks=ticks[ticks.chromosome.isin(['chr1','chr5','chr9','chr13','chr17','chr21','chrX'])]
ax[0].set_xticks(ticks.midpoint,[x.replace('chr','') for x in ticks.chromosome])
ax[0].set(xlabel='Chromosome',ylabel='−log10 cross-tissue P',title='Cross-tissue methylation')
ax[0].set_ylim(0,float(meta.minus_log10_p.max())+2)
ax[0].margins(x=.005)

order=tissue[['window','meta_pvalue']].drop_duplicates().sort_values('meta_pvalue').window.tolist()
mat=tissue.pivot(index='window',columns='tissue',values='delta_beta').reindex(index=order,columns=m.TISSUE_ORDER)
lim=max(float(np.nanquantile(np.abs(mat.to_numpy()),.97)),.05)
im=ax[1].imshow(mat,cmap=m.DIVERGING_OKABE,norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto')
ax[1].set_xticks(range(4),[x.capitalize() for x in mat.columns],rotation=60,ha='right')
ax[1].set_yticks(range(12),mat.index)
ax[1].set_title('Top-window effects',pad=12)
cb=fig.colorbar(im,cax=fig.add_axes([.915,.79,.012,.10])); cb.set_label('Δβ',labelpad=1); cb.ax.tick_params(labelsize=8,length=2)

y=-np.log10(np.clip(signals.signal_acat_pvalue,1e-300,1))
ax[2].scatter(signals.median_weighted_delta_beta,y,s=np.clip(5+5*signals.gene_count,5,22),c=c['sky_blue'],alpha=.38,linewidths=0,rasterized=True)
sig_signal=signals.signal_fdr.lt(.05)
if sig_signal.any(): ax[2].scatter(signals.loc[sig_signal,'median_weighted_delta_beta'],y[sig_signal],s=20,c=c['vermillion'])
ax[2].set(xlabel='Median cross-tissue Δβ',ylabel='−log10 promoter-signal P',title='Collapsed promoter signals')

mat=modules.pivot(index='module',columns='target_state',values='spearman_rho').reindex(index=m.MODULE_ORDER,columns=m.STATE_ORDER)
cmap=m.DIVERGING_OKABE.copy(); cmap.set_bad('#E6E6E6')
im=ax[3].imshow(mat,cmap=cmap,norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),aspect='auto')
ax[3].set_xticks(range(4),mat.columns,rotation=60,ha='right')
ax[3].set_yticks(range(9),[textwrap.fill(x.replace('Host-response circuit','Host-response program'),22,break_long_words=False,break_on_hyphens=False) for x in mat.index])
ax[3].set_title('Program correlations',pad=12)
cb=fig.colorbar(im,cax=fig.add_axes([.915,.47,.012,.10])); cb.set_label('Spearman ρ',labelpad=1); cb.ax.tick_params(labelsize=8,length=2)
for a in [ax[1],ax[3]]:
    a.tick_params(length=0,pad=2)
    for s in a.spines.values(): s.set_visible(False)

markers={'control':'o','nec':'^','future_nec':'D'}
labels={'control':'Non-NEC','nec':'NEC','future_nec':'Pre-NEC'}
for cond in labels:
    d=pca[pca.analysis_condition.eq(cond)]
    ax[4].scatter(d.PC1,d.PC2,s=30,marker=markers[cond],c=m.CONDITION_COLORS[cond],edgecolors='black',linewidths=.4,label=labels[cond])
ax[4].set(xlabel=f'PC1 ({pca.PC1_variance.iloc[0]*100:.1f}%)',ylabel=f'PC2 ({pca.PC2_variance.iloc[0]*100:.1f}%)',title='Residual stool methylome')
ax[4].legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=2,columnspacing=.8,handletextpad=.3)
primary=scores[scores.analysis_condition.isin(['control','nec'])]
rng=np.random.default_rng(20261106)
for pos,cond in enumerate(['control','nec']):
    d=primary[primary.analysis_condition.eq(cond)]
    v=d.leave_one_subject_out_score.to_numpy()
    ax[5].scatter(pos+rng.uniform(-.07,.07,len(v)),v,s=34,marker=markers[cond],c=m.CONDITION_COLORS[cond],edgecolors='black',linewidths=.4,zorder=3)
    q=np.quantile(v,[.25,.5,.75]); ax[5].vlines(pos,q[0],q[2],color='black',lw=1.5); ax[5].hlines(q[1],pos-.12,pos+.12,color='black',lw=1)
auc=roc_auc_score(primary.analysis_condition.eq('nec'),primary.leave_one_subject_out_score)
ax[5].set(xticks=[0,1],xticklabels=['Non-NEC','NEC'],xlim=(-.4,1.4),ylabel='Leave-one-subject-out\nprojection score',title='Adjusted stool projection')
ax[5].text(.03,.97,f'AUC {auc:.2f}',transform=ax[5].transAxes,ha='left',va='top',fontsize=8)

for i,a in enumerate(ax):
    a.text(-.26 if i in [1,3] else -.18,1.08,chr(65+i),transform=a.transAxes,fontweight='bold',fontsize=10,va='bottom')
    a.grid(False)
fig.canvas.draw()
renderer=fig.canvas.get_renderer(); outside=[]
for t in fig.findobj(Text):
    if t.get_visible() and t.get_text():
        b=t.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        if b.x0<0 or b.y0<0 or b.x1>1 or b.y1>1: outside.append({'text':t.get_text(),'bbox':list(b.bounds)})
stem=out/'Supplementary Figure 10'
fig.savefig(str(stem)+'.pdf',dpi=600)
fig.savefig(str(stem)+'.png',dpi=600)
with Image.open(str(stem)+'.png') as im:
    rgb=im.convert('RGB'); rgb.save(str(stem)+'.png',dpi=(600,600)); rgb.save(str(stem)+'.tiff',compression='tiff_lzw',dpi=(600,600))
audit={'width_mm':180,'height_mm':275,'font':'Liberation Sans','minimum_font_pt':min(t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text()),'outside_text':outside,'cross_tissue_window_tests':len(meta),'cross_tissue_window_fdr_0_05':len(sig),'promoter_signals':len(signals),'promoter_signal_fdr_0_05':int(sig_signal.sum()),'source_stage':'final covariate-refresh and promoter-collapse outputs'}
(out/'Supplementary Figure 10_layout_audit.json').write_text(json.dumps(audit,indent=2))
legend='''# Supplementary Figure 10 | Cross-tissue methylation and stool projection

**A**, Cross-tissue P values for 890,377 1-kb windows represented in at least two tissue sources. Vermillion points identify the two windows with Benjamini–Hochberg (BH) false discovery rate (FDR) <0.05 in the final covariate-refreshed analysis. **B**, Tissue-specific methylation differences for the 12 smallest cross-tissue P values; positive Δβ denotes higher methylation in NEC. Stool estimates adjust for corrected gestational age, postnatal day, sex and race. **C**, All 41,280 genomic promoter signals formed by collapsing overlapping gene promoter annotations. Point area represents the number of overlapping annotations; no collapsed signal has BH FDR <0.05. **D**, Spearman correlations between negative promoter methylation effects and intestinal cell-state expression changes for nine named programs. **E**, Principal components of 500 covariate-residualized stool methylation windows; pre-NEC observations are displayed separately. **F**, Nested leave-one-subject-out projection scores for 13 primary stool subjects (eight NEC and five non-NEC); feature selection, residualization and axis estimation omit the held-out subject. Bars show medians and interquartile ranges. The area under the receiver operating characteristic curve is 0.70.

Gray cells in D denote unavailable correlations. The four sources comprise 80 samples and 79 source-specific subject identifiers. Ileum and colon non-NEC comparators were healed NEC at reanastomosis. Cross-source subject linkage is unavailable; weighted Stouffer integration is an exploratory cross-tissue screen under conditional independence. Within-tissue tests, cross-tissue windows, promoter genes, collapsed signals and module correlations use separate declared BH families. The final within-tissue window families each contain zero FDR <0.05 results. Liberation Sans and the Okabe–Ito colorblind-friendly palette are used throughout.
'''
(ROOT/'publication_figures/Supplementary_Figure_10_legend.md').write_text(legend,encoding='utf-8')
print(json.dumps(audit))
