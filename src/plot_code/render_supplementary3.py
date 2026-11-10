"""Refresh Supplementary Figure 3 from the complete cloud count-model outputs."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from render_revision_figures import canvas,panel,clean,heat,finish,text_box,OI,CMAP,STATES

STATES=[st for st in STATES if st!='epithelial_stem']
LABELS=['M cells','LECs','Tregs','T/NK','Epithelial','Myeloid','Vascular EC','Fibroblasts']

def label_points(ax,data,x,y,n):
    candidates=data[data.q.lt(.05)].sort_values('q')
    if candidates.empty:candidates=data.sort_values('q').head(1)
    selected=[];span=max(1.,float(data[x].max()-data[x].min()))
    for _,row in candidates.iterrows():
        if all(abs(row[x]-r[x])>.08*span for r in selected):selected.append(row)
        if len(selected)>=n:break
    ax.margins(x=.20,y=.35);ax.set_ylim(bottom=0);ax.figure.canvas.draw();r=ax.figure.canvas.get_renderer();frame=ax.get_window_extent(r);used=[]
    for row in selected:
        placed=False
        for dx,dy in [(dx,dy) for dy in [5,15,25,-10,-20,-30,35] for dx in [4,-4,16,-16,28,-28,40,-40]]:
            t=ax.annotate(str(row.gene),(row[x],row[y]),xytext=(dx,dy),textcoords='offset points',fontsize=8,ha='left' if dx>0 else 'right',arrowprops={'arrowstyle':'-','lw':.3,'color':'black'},annotation_clip=False)
            b=text_box(t,r)
            if b.x0>=frame.x0+1 and b.x1<=frame.x1-1 and b.y0>=frame.y0+1 and b.y1<=frame.y1-1 and not any(b.expanded(1.06,1.08).overlaps(z) for z in used):used.append(b);placed=True;break
            t.remove()
        if not placed:raise RuntimeError('Could not place gene label '+str(row.gene))

def volcano(ax,frame,fc,q,n=3):
    d=frame[['gene',fc,q]].rename(columns={fc:'effect',q:'q'}).copy()
    if not np.isfinite(d[['effect','q']].to_numpy(float)).all():raise ValueError('Nonfinite volcano input')
    d['minus_log10_fdr']=-np.log10(d.q.clip(lower=1e-300));d['class']=np.where(d.q.ge(.05),'FDR ≥ 0.05',np.where(d.effect.gt(0),'NEC higher','NEC lower'))
    for name,color in [('FDR ≥ 0.05','#B9B9B9'),('NEC lower',OI['blue']),('NEC higher',OI['red'])]:
        z=d[d['class'].eq(name)];ax.scatter(z.effect,z.minus_log10_fdr,s=3 if name=='FDR ≥ 0.05' else 7,color=color,alpha=.35 if name=='FDR ≥ 0.05' else .85,linewidths=0,rasterized=True)
    ax.set_xlabel('Log2 fold change');ax.set_ylabel('−log10 FDR');clean(ax);label_points(ax,d,'effect','minus_log10_fdr',n)
    return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);s=a.results/'04_corrected_results'/'02_statistical_reanalysis';records=[]
    def read(name):
        path=s/name;d=pd.read_csv(path);records.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(d)});return d
    all_de=read('single_cell_pseudobulk_edger_all_cells.csv')
    state_frames=[]
    for st in STATES:state_frames.append(read('single_cell_pseudobulk_target_genes_'+st+'.csv'))
    de=pd.concat(state_frames,ignore_index=True)
    logcpm=read('single_cell_pseudobulk_target_logcpm_all_cells.csv')
    b1=read('bulk_limma_GSE46619_nec_vs_control.csv');b2=read('bulk_limma_GSE64801_nec_vs_control.csv')
    cov_path=a.results/'new_results'/'donor_covariates.csv';cov=pd.read_csv(cov_path);records.append({'file':str(cov_path),'sha256':hashlib.sha256(cov_path.read_bytes()).hexdigest(),'rows':len(cov)})
    if cov.donor_id.nunique()!=11 or len(logcpm)!=31:raise ValueError('Unexpected donor or response-gene denominator')
    out=a.output/'source_data'/'Supplementary_Figure_3';out.mkdir(parents=True,exist_ok=True)
    fig=canvas(260)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=n,ms=3) for n,c in [('FDR ≥ 0.05','#B9B9B9'),('NEC lower',OI['blue']),('NEC higher',OI['red'])]],loc='upper left',bbox_to_anchor=(.08,.995),ncol=3,columnspacing=1.1,handletextpad=.3)
    aa=panel(fig,[.13,.64,.27,.31],'A','All-cell donor expression');va=volcano(aa,all_de,'logFC','FDR',3);va.to_csv(out/'Supplementary_Figure_3A.csv',index=False)
    bb=panel(fig,[.64,.60,.27,.35],'B','State-resolved gene effects')
    gene_order=de.groupby('gene').logFC.apply(lambda x:x.abs().max()).sort_values(ascending=False).index.tolist()
    xs={x:i for i,x in enumerate(STATES)};ys={x:i for i,x in enumerate(gene_order)}
    sizes=4+20*np.clip(-np.log10(de.FDR.clip(lower=1e-8)),0,3)
    bb.scatter(de.cell_state.map(xs),de.gene.map(ys),c=de.logFC,s=sizes,cmap=CMAP,vmin=-4,vmax=4,edgecolor='black',linewidth=.2)
    bb.set_xticks(range(len(STATES)),LABELS,rotation=90);bb.set_yticks(range(len(gene_order)),gene_order);bb.set_xlim(-.6,len(STATES)-.4);bb.set_ylim(len(gene_order)-.5,-.5);bb.tick_params(length=0,pad=2)
    for spine in bb.spines.values():spine.set_visible(False)
    bc=fig.add_axes([.94,.655,.012,.20]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-4,4),cmap=CMAP),cax=bc);cb.set_label('Log2 fold change',labelpad=3);cb.ax.tick_params(labelsize=8,length=2)
    de.to_csv(out/'Supplementary_Figure_3B.csv',index=False)
    cond={f'D{int(x.donor_id)}':str(x.condition) for x in cov.itertuples()}
    donor_order=[d for d in logcpm.columns[1:] if cond[d]=='control']+[d for d in logcpm.columns[1:] if cond[d]=='nec']
    m=logcpm.set_index('gene')[donor_order];m=((m.T-m.mean(axis=1))/m.std(axis=1,ddof=0).replace(0,np.nan)).T.fillna(0).T
    cc=panel(fig,[.21,.30,.69,.14],'C','Donor expression of 31 response genes');m.index=[d[1:]+' '+('NEC' if cond[d]=='nec' else 'Control') for d in m.index];heat(fig,cc,m,-2.2,2.2,'Gene z score',90);m.to_csv(out/'Supplementary_Figure_3C.csv')
    dd=panel(fig,[.12,.045,.235,.13],'D','GSE46619');vd=volcano(dd,b1,'logfc','fdr_bh',3);vd.to_csv(out/'Supplementary_Figure_3D.csv',index=False)
    ee=panel(fig,[.44,.045,.235,.13],'E','GSE64801');ve=volcano(ee,b2,'logfc','fdr_bh',1);ve.to_csv(out/'Supplementary_Figure_3E.csv',index=False)
    ff=panel(fig,[.77,.045,.20,.13],'F','Effect concordance');concordance=[]
    for name,bulk,color in [('GSE46619',b1,OI['orange']),('GSE64801',b2,OI['green'])]:
        z=all_de[['gene','logFC']].merge(bulk[['gene','logfc']],on='gene').rename(columns={'logFC':'single_cell_logfc','logfc':'bulk_logfc'});z['accession']=name
        rho=float(z[['single_cell_logfc','bulk_logfc']].rank(method='average').corr().iloc[0,1]);z['spearman_rho']=rho
        ff.scatter(z.single_cell_logfc,z.bulk_logfc,s=2,color=color,alpha=.17,linewidths=0,rasterized=True,label=f'{name}: ρ={rho:.2f}')
        slope,intercept=np.polyfit(z.single_cell_logfc,z.bulk_logfc,1);xr=np.linspace(z.single_cell_logfc.quantile(.01),z.single_cell_logfc.quantile(.99),100);ff.plot(xr,slope*xr+intercept,color=color,lw=.8);concordance.append(z)
    ff.set_xlabel('Single-cell log2 FC');ff.set_ylabel('Bowel log2 FC');clean(ff);ff.legend(loc='upper left',handlelength=.4,handletextpad=.2,labelspacing=.3,borderaxespad=.15)
    pd.concat(concordance,ignore_index=True).to_csv(out/'Supplementary_Figure_3F.csv',index=False)
    finish(fig,'Supplementary Figure 3',SimpleNamespace(out=a.output))
    (a.output/'supplementary3_inputs.json').write_text(json.dumps({'input_files':records,'all_tested_genes_A':len(all_de),'state_gene_rows_B':len(de),'gene_rows_C':len(logcpm),'donors_C':len(donor_order),'display':'all complete model rows; no point sampling'},indent=2),encoding='utf-8')
    print(a.output/'Supplementary Figure 3.png')

if __name__=='__main__':main()
