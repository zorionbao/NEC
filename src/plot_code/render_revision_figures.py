# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# Scatter: PCA -> parameter inherit; composition: BarComposition -> parameter inherit.
# Intervals: PairedBoxScatter -> parameter inherit; heatmap/dots -> parameter inherit.
# Donor/infant source schemas differ from those of the production assets.
import argparse, hashlib, json, textwrap
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans'],
 'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
 'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
 'xtick.direction':'out','ytick.direction':'out','legend.frameon':False,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':300,
 'figure.facecolor':'white','axes.facecolor':'white'})
import matplotlib.pyplot as plt
from matplotlib import font_manager
for _font in (Path(__file__).resolve().parent/'fonts').glob('LiberationSans-*.ttf'):
 font_manager.fontManager.addfont(str(_font))
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
OI={'black':'#000000','orange':'#E69F00','sky':'#56B4E9','green':'#009E73','yellow':'#F0E442','blue':'#0072B2','red':'#D55E00','purple':'#CC79A7','white':'#FFFFFF'}
GROUP={'control':OI['blue'],'nec':OI['red']}
STATES=['M_cell','LEC','T_reg','T_NK','epithelial','epithelial_stem','myeloid','vascular_endothelial','fibroblast']
LABEL=dict(zip(STATES,['M cells','Lymphatic endothelial','Regulatory T cells','T and NK cells','Epithelial','Epithelial stem','Myeloid','Vascular endothelial','Fibroblast']))
COLOR=dict(zip(STATES,[OI['orange'],OI['sky'],OI['green'],OI['blue'],OI['red'],OI['white'],OI['purple'],OI['black'],OI['yellow']]))
TCOLOR={'M_cell':OI['orange'],'LEC':OI['green'],'T_reg':OI['blue']}
CMAP=LinearSegmentedColormap.from_list('Okabe-Ito signed',[OI['blue'],'white',OI['red']])
SEQ=LinearSegmentedColormap.from_list('Okabe-Ito positive',['white',OI['sky'],OI['blue']])
AXES=['CXCL16-CXCR6','IL22-IL22RA1','EFNA1-EPHA2','VEGFC-FLT4','TNF-ICAM1','CCL21-CCR7']
HGENES=['HCAR2','CCL3','TNIP3','TNFAIP6','IL1B','CCL20','CSF2','CCL21','CXCL1','VCAM1','SELE','ICAM1','CXCL16','CXCR6','FOXP3','TIGIT','MUC2','OLFM4']
TGENES=['GP2','SPIB','PROX1','ACKR4','FOXP3','TIGIT','CXCR6']
BGENES=['CXCL16','CCL21','FLT4','LYVE1','IL22RA1','MUC2','OLFM4','FOXP3','TIGIT','ICAM1','IL1B','CXCL8']
def cond(x):return 'nec' if str(x).lower().startswith('nec') else 'control'
def state(x):return {'M cell':'M_cell','M cells':'M_cell','M cell + LEC':'combined','Treg':'T_reg','Tregs':'T_reg','Lymphatic endothelial cells':'LEC'}.get(str(x),str(x))
def col(d,options,required=True):
 for c in options:
  if c in d.columns:return c
 if required:raise KeyError(f'Missing {options}; got {d.columns.tolist()}')
def sc(d):return col(d,['target_state','cell_state','state'])
def order(d):
 dc=col(d,['donor_id','donor','id_number']);x=d[[dc,'condition']].drop_duplicates();return x.sort_values(['condition',dc])[dc].tolist()
class Inputs:
 def __init__(self,m,o):
  self.path=Path(m);self.m=json.loads(self.path.read_text(encoding='utf-8'));self.out=Path(o);self.out.mkdir(parents=True,exist_ok=True);self.sources=[];self.cache={}
 def get(self,k):
  if k not in self.cache:
   v=self.m['inputs'][k];p=Path(v['path'] if isinstance(v,dict) else v)
   if not p.is_absolute():p=self.path.parent/p
   d=pd.read_csv(p)
   if 'condition' in d:d['condition']=d['condition'].map(cond)
   for c in ['target_state','cell_state','state']:
    if c in d:d[c]=d[c].map(state)
   self.sources.append({'key':k,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(d),'columns':d.columns.tolist()});self.cache[k]=d
  return self.cache[k].copy()
 def save(self,f,p,d):
  path=self.out/'source_data'/f'Figure_{f}';path.mkdir(parents=True,exist_ok=True);d.to_csv(path/f'Figure_{f}{p}.csv',index=False)
def canvas(h=250):return plt.figure(figsize=(180/25.4,h/25.4))
def panel(fig,box,letter,title):
 a=fig.add_axes(box);a.set_title(title,loc='left',pad=8);fig.text(max(.01,box[0]-.052),box[1]+box[3]+.015,letter,weight='bold',fontsize=11,va='bottom');return a
def clean(a):a.grid(False);a.tick_params(pad=2,length=2.5);a.spines[['top','right']].set_visible(False)
def text_box(t,renderer):
 if isinstance(t,mpl.text.Annotation):t.update_positions(renderer);return mpl.text.Text.get_window_extent(t,renderer)
 return t.get_window_extent(renderer)
def glegend(fig):
 fig.legend(handles=[Line2D([],[],marker='o',ls='',color=GROUP[g],label=l,ms=4) for g,l in [('control','Control'),('nec','NEC')]],loc='upper right',bbox_to_anchor=(.985,.994),ncol=2,handletextpad=.3,columnspacing=1)
def heat(fig,a,m,vmin=-2,vmax=2,label='Z score',rotation=90,cmap=CMAP):
 im=a.imshow(m.to_numpy(float),cmap=cmap,vmin=vmin,vmax=vmax,aspect='auto',interpolation='nearest');a.set_xticks(range(m.shape[1]),m.columns,rotation=rotation,ha='center');a.set_yticks(range(m.shape[0]),[str(x).replace('D','') for x in m.index]);a.tick_params(length=0,pad=3)
 for s in a.spines.values():s.set_visible(False)
 b=a.get_position();ca=fig.add_axes([b.x1+.015,b.y0,.012,b.height]);cb=fig.colorbar(im,cax=ca);cb.set_label(label,fontsize=8,labelpad=4);cb.ax.tick_params(labelsize=8,length=2)
def forest(a,d,ef='hedges_g',lo='hedges_g_ci_low',hi='hedges_g_ci_high',label='feature',colors=None,xlabel='Hedges g, NEC versus control',limits=None):
 d=d.reset_index(drop=True);cs=colors if colors is not None else [OI['red'] if x>=0 else OI['blue'] for x in d[ef]]
 if lo in d and hi in d:
  for j,r in d.iterrows():
   low,high=r[lo],r[hi]
   if limits:
    low,high=max(low,limits[0]),min(high,limits[1])
    for bound,tip in [(limits[0],r[lo]<limits[0]),(limits[1],r[hi]>limits[1])]:
     if tip:a.annotate('',xy=(bound,j),xytext=(bound+(.18 if bound==limits[0] else -.18),j),arrowprops={'arrowstyle':'-|>','lw':.8,'color':cs[j]})
   a.plot([low,high],[j,j],color=cs[j],lw=1.2)
 a.scatter(d[ef],range(len(d)),c=cs,s=19,zorder=3,edgecolors='white',lw=.3);a.set_yticks(range(len(d)),d[label].astype(str));a.invert_yaxis();a.set_xlabel(xlabel);a.axvline(0,color='black',lw=.5,ls=':');clean(a);a.margins(x=.12,y=.05)
 if limits:a.set_xlim(limits)
def points(a,d,vc,ylabel=''):
 for i,g in enumerate(['control','nec']):
  t=d[d['condition'].map(cond)==g];a.scatter(i+np.linspace(-.1,.1,len(t)),t[vc],s=22,c=GROUP[g],edgecolor='white',lw=.35,zorder=3)
  if len(t):a.plot([i-.2,i+.2],[t[vc].mean()]*2,color=GROUP[g],lw=1.5)
 a.set_xticks([0,1],['Control','NEC']);a.set_xlim(-.45,1.45);a.set_ylabel(ylabel);clean(a)
def native_vascular_points(a,d):
 if 'eligible' in d:d=d[d.eligible.astype(str).str.lower().isin(['true','1'])].copy()
 assert not d.duplicated(['donor_id','gene']).any()
 labels=[]
 for index,gene in enumerate(['SELE','VCAM1']):
  part=d[d.gene.eq(gene)];ns={}
  for offset,group in enumerate(['control','nec']):
   z=part[part.condition.eq(group)].copy();ns[group]=z.donor_id.nunique();assert ns[group]==(3 if group=='control' else 5)
   xpos=index*3+offset;a.scatter(xpos+np.linspace(-.13,.13,len(z)),z.log2_cpm,s=22,color=GROUP[group],edgecolor='white',lw=.35,zorder=3);a.plot([xpos-.24,xpos+.24],[z.log2_cpm.mean()]*2,color=GROUP[group],lw=1.3)
  labels.append(gene+chr(10)+str(ns['control'])+' control / '+str(ns['nec'])+' NEC')
 a.set_xticks([.5,3.5],labels);a.set_xlim(-.6,4.6);a.set_ylabel('TMM log2 CPM');clean(a);return d
def label_donors(a,d,xc,yc,dc):
 d=d.sort_values([yc,xc],ascending=[False,True]);a.margins(y=.18);a.figure.canvas.draw();renderer=a.figure.canvas.get_renderer();frame=a.get_window_extent(renderer);occupied=[];coords=a.transData.transform(np.column_stack([d[xc].to_numpy(float)*100,d[yc].to_numpy(float)]))
 candidates=[(dx,dy) for dy in [3,14,-8,25,-19,36,-30,47,-41] for dx in [4,-25,17,-38]]
 for row,xy in zip(d.to_dict('records'),coords):
  found=False
  for dx,dy in candidates:
   t=a.annotate(str(row[dc]).replace('D',''),(row[xc]*100,row[yc]),xytext=(dx,dy),textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'-','lw':.3,'color':'black'},annotation_clip=False)
   box=text_box(t,renderer);inside=box.x0>=frame.x0+1 and box.y0>=frame.y0+1 and box.x1<=frame.x1-1 and box.y1<=frame.y1-1
   overlap=any(box.expanded(1.08,1.12).overlaps(old) for old in occupied)
   covers_point=any(box.x0-2<=x<=box.x1+2 and box.y0-2<=y<=box.y1+2 for x,y in coords)
   if inside and not overlap and not covers_point:occupied.append(box);found=True;break
   t.remove()
  if not found:raise RuntimeError('Could not place donor label '+str(row[dc]))
def dots(fig,a,d,row,cl,ef,q,columns,rows,label='Log2 fold change',lim=3):
 d=d[d[cl].isin(columns)&d[row].isin(rows)].copy();xs={x:i for i,x in enumerate(columns)};ys={x:i for i,x in enumerate(rows)};sizes=8+33*np.clip(-np.log10(np.maximum(d[q].to_numpy(float),1e-8))/2,0,1)
 a.scatter(d[cl].map(xs),d[row].map(ys),c=d[ef],s=sizes,cmap=CMAP,vmin=-lim,vmax=lim,edgecolor='black',lw=.25);a.set_xticks(range(len(columns)),columns,rotation=50,ha='right');a.set_yticks(range(len(rows)),rows);a.set_xlim(-.6,len(columns)-.4);a.set_ylim(len(rows)-.5,-.5);a.tick_params(length=0)
 for s in a.spines.values():s.set_visible(False)
 b=a.get_position();ca=fig.add_axes([b.x1+.015,b.y0,.012,b.height*.65]);cb=fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-lim,lim),cmap=CMAP),cax=ca);cb.set_label(label,fontsize=8);cb.ax.tick_params(labelsize=8)
def finish(fig,n,inp):
 fig.canvas.draw();r=fig.canvas.get_renderer();bad=[];texts=[];hidden_tick_text=set()
 for a in fig.axes:
  for axis in [a.xaxis,a.yaxis]:
   lower,upper=sorted(axis.get_view_interval())
   for tick in axis.get_major_ticks()+axis.get_minor_ticks():
    if tick.get_loc()<lower-1e-9 or tick.get_loc()>upper+1e-9:hidden_tick_text.update([id(tick.label1),id(tick.label2)])
 for t in fig.findobj(mpl.text.Text):
  if not t.get_visible() or not t.get_text() or id(t) in hidden_tick_text:continue
  b=text_box(t,r)
  texts.append((t,b))
  if t.get_fontsize()<7.99:bad.append({'text':t.get_text(),'reason':'font below 8 pt'})
  if b.x0< -1 or b.y0< -1 or b.x1>fig.bbox.width+1 or b.y1>fig.bbox.height+1:bad.append({'text':t.get_text(),'reason':'outside canvas'})
 if bad:
  (inp.out/f'Figure_{n}_layout_issues.json').write_text(json.dumps(bad,indent=2),encoding='utf-8');raise RuntimeError(f'Figure {n}: {len(bad)} layout defects')
 issue_file=inp.out/f'Figure_{n}_layout_issues.json'
 if issue_file.exists():issue_file.unlink()
 overlaps=[]
 for i,(t,b) in enumerate(texts):
  for u,c in texts[i+1:]:
   w=max(0,min(b.x1,c.x1)-max(b.x0,c.x0));h=max(0,min(b.y1,c.y1)-max(b.y0,c.y0))
   if w*h>.15*min(b.width*b.height,c.width*c.height):overlaps.append({'text_1':t.get_text(),'text_2':u.get_text(),'overlap_pixels':w*h})
 (inp.out/f'Figure_{n}_layout_audit.json').write_text(json.dumps({'minimum_font_pt':min(t.get_fontsize() for t,b in texts),'canvas_width_mm':fig.get_figwidth()*25.4,'canvas_height_mm':fig.get_figheight()*25.4,'outside_canvas':bad,'text_overlap_candidates':overlaps},indent=2),encoding='utf-8')
 p=inp.out/(f'Figure {n}' if isinstance(n,int) else str(n));fig.savefig(str(p)+'.pdf',dpi=300);fig.savefig(str(p)+'.png',dpi=300);fig.savefig(str(p)+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'});plt.close(fig)
 for ext,dpi in [('.png',300),('.tiff',600)]:
  with Image.open(str(p)+ext) as im:rgb=im.convert('RGB')
  rgb.save(str(p)+ext,dpi=(dpi,dpi),**({'compression':'tiff_lzw'} if ext=='.tiff' else {}))
def figure1(inp):
 fig=canvas(250);glegend(fig)
 a=panel(fig,[.10,.785,.36,.16],'A','Neonatal intestinal atlas');d=inp.get('atlas');s=sc(d);ux=col(d,['umap_1','UMAP1']);uy=col(d,['umap_2','UMAP2'])
 for st in STATES:
  t=d[d[s]==st];a.scatter(t[ux],t[uy],s=.35,color=COLOR[st],linewidths=.06 if st=='epithelial_stem' else 0,edgecolors='black' if st=='epithelial_stem' else None,rasterized=True)
 a.set_xlabel('UMAP 1');a.set_ylabel('UMAP 2');a.set_xticks([]);a.set_yticks([]);clean(a);inp.save(1,'A',d)
 a=panel(fig,[.61,.785,.35,.16],'B','Independent-donor composition');d=inp.get('composition');dc=col(d,['donor_id','donor']);fc=col(d,['fraction','posterior_fraction_all_cells']);m=d.pivot_table(index=dc,columns=sc(d),values=fc).reindex(index=order(d),columns=STATES).fillna(0);left=np.zeros(len(m))
 for s in STATES:
  a.barh(range(len(m)),m[s],left=left,color=COLOR[s],edgecolor='black' if s=='epithelial_stem' else 'none',linewidth=.3);left+=m[s].to_numpy()
 a.set_yticks(range(len(m)),[str(x).replace('D','') for x in m.index]);a.invert_yaxis();a.set_xlabel('Fraction of captured cells');a.set_xlim(0,1);clean(a)
 dg=d[[dc,'condition']].drop_duplicates().set_index(dc)['condition']
 for tick,x in zip(a.get_yticklabels(),m.index):tick.set_color(GROUP[cond(dg[x])])
 inp.save(1,'B',d)
 fig.legend(handles=[Patch(facecolor=COLOR[s],edgecolor='black' if s=='epithelial_stem' else 'none',label=LABEL[s]) for s in STATES],ncol=3,loc='upper center',bbox_to_anchor=(.51,.744),columnspacing=1.2,handlelength=1,handletextpad=.4)
 a=panel(fig,[.12,.525,.33,.12],'C','Developmental and surgical ages');d=inp.get('donor_covariates');ga=col(d,['gestational_age_weeks','GA']);sa=col(d,['age_at_surgery_days','surgery_age_days'])
 for g,t in d.groupby('condition'):a.scatter(t[ga],t[sa],s=25,color=GROUP[g],edgecolor='white',lw=.4)
 a.set_xlabel('Gestational age (weeks)');a.set_ylabel('Age at surgery (days)');clean(a);inp.save(1,'C',d)
 a=panel(fig,[.62,.525,.34,.12],'D','Held-out target-state annotation');d=inp.get('target_validation');d=d.set_index(sc(d)).reindex(['M_cell','LEC','T_reg'])
 for k,(metric,color) in enumerate([('precision',OI['blue']),('recall',OI['orange']),('f1',OI['green'])]):
  mc=col(d,[metric,'F1' if metric=='f1' else metric]);a.bar(np.arange(3)+(k-1)*.24,d[mc],width=.22,color=color,label='F1' if metric=='f1' else metric.capitalize())
 a.set_xticks(range(3),['M cells','LECs','Tregs']);a.set_ylim(0,1.05);a.set_ylabel('Classification score');clean(a);a.legend(loc='upper center',bbox_to_anchor=(.5,-.27),ncol=3,columnspacing=.6,handlelength=.9,handletextpad=.25);inp.save(1,'D',d.reset_index())
 a=panel(fig,[.38,.275,.58,.17],'E','Donor-level host-state effects');d=inp.get('host_effects');lc=col(d,['display_label','feature']);d['display']=d[lc].str.replace('Regulatory T cells','Treg',regex=False).str.replace('Lymphatic endothelial','LEC',regex=False).str.replace('T and NK cells','T/NK',regex=False);forest(a,d,label='display');inp.save(1,'E',d)
 a=panel(fig,[.13,.09,.74,.12],'F','Raw-count pseudobulk response genes');d=inp.get('host_gene_zscores');dc=col(d,['donor_id','donor']);genes=[g for g in HGENES if g in d];m=d.set_index(dc).reindex(order(d))[genes];heat(fig,a,m,label='Donor z score');inp.save(1,'F',d[[dc,'condition']+genes]);finish(fig,1,inp)

def figure2(inp):
 fig=canvas(275);glegend(fig)
 a=panel(fig,[.29,.81,.22,.13],'A','Predicted pathway activity');d=inp.get('axis_effects');d=d.set_index('feature').reindex(AXES).reset_index();forest(a,d);inp.save(2,'A',d)
 a=panel(fig,[.69,.81,.19,.13],'B','Donor pathway scores');d=inp.get('axis_scores');dc=col(d,['donor_id','donor']);ac=col(d,['axis','feature']);vc=col(d,['axis_score','score']);m=d.pivot_table(index=dc,columns=ac,values=vc).reindex(index=order(d),columns=AXES);heat(fig,a,m,-2,2,'Axis z score',90);inp.save(2,'B',d)
 a=panel(fig,[.13,.56,.35,.13],'C','Matched-receptor specificity');d=inp.get('lr_specificity');vc=col(d,['effect','hedges_g','score']);tc=col(d,['kind','type','is_observed']);obs=d[tc].astype(str).str.lower().isin(['observed','true','1']);a.hist(d.loc[~obs,vc].dropna(),bins=25,color=OI['sky'],edgecolor='white',lw=.25)
 for v in d.loc[obs,vc].dropna():a.axvline(v,color=OI['red'],lw=1.5)
 a.set_xlabel('CCL21-paired pathway effect');a.set_ylabel('Matched receptors');clean(a);a.legend(handles=[Patch(facecolor=OI['sky'],label='Receptor null'),Line2D([],[],color=OI['red'],label='CCR7')],loc='upper center',bbox_to_anchor=(.5,-.35),ncol=2,handlelength=1,columnspacing=.8);inp.save(2,'C',d)
 a=panel(fig,[.69,.55,.19,.15],'D','Broad-state gene effects');d=inp.get('state_de');d['display_state']=d[sc(d)].map(LABEL).fillna(d[sc(d)]);rows=['CSF2','CCL21','CXCL1','VCAM1','SELE','IL1B','CCL20','CCL3','TNIP3','TNFAIP6','TIGIT','CXCR6'];dots(fig,a,d,'gene','display_state','logFC',col(d,['FDR','fdr_bh']),[LABEL[s] for s in ['M_cell','LEC','T_reg','T_NK','epithelial','myeloid','vascular_endothelial']],rows,lim=4);a.set_xticklabels(['M cells','LECs','Tregs','T/NK','Epithelial','Myeloid','Vascular EC'],rotation=90,ha='center');inp.save(2,'D',d)
 a=panel(fig,[.13,.31,.35,.13],'E','Composition-adjusted bowel expression');d=inp.get('bulk_de');cc=col(d,['contrast_label','comparison','contrast']);d['contrast_display']=[str(a)+'\nNEC/'+('SIP' if 'sip' in str(c).lower() else 'control') for a,c in zip(d['accession'],d[cc])];co=d['contrast_display'].drop_duplicates().tolist();genes=[x for x in BGENES if x in set(d['gene'])];m=d.pivot_table(index='contrast_display',columns='gene',values='logFC').reindex(index=co,columns=genes);m.index=[x.replace('GSE46619 ','46619\n').replace('GSE64801 ','64801\n').replace(' versus ','/').replace('NEC_vs_','NEC/') for x in m.index];heat(fig,a,m,-4,4,'Log2 fold change');inp.save(2,'E',d)
 a=panel(fig,[.68,.31,.28,.13],'F','LYVE1-region tissue injury');d=inp.get('imaging_subjects');vc=col(d,['mean_double_positive_cells_per_hpf','mean_count','mean_double_positive_cells_per_image']);points(a,d,vc,'TUNEL-positive nuclei / image');hp=inp.get('imaging_fields');hc=col(hp,['double_positive_cells_per_hpf','double_positive_cells_per_image','count'])
 for i,g in enumerate(['control','nec']):
  t=hp[hp['condition'].map(cond)==g];a.scatter(i+np.linspace(-.17,.17,len(t)),t[hc],s=15,facecolor='none',edgecolor=GROUP[g],lw=.6,zorder=1)
 inp.save(2,'F',d);inp.save(2,'F_fields',hp)
 a=panel(fig,[.13,.075,.35,.13],'G','Native vascular markers');d=inp.get('native_vascular_markers');d=native_vascular_points(a,d);inp.save(2,'G',d)
 a=panel(fig,[.68,.075,.28,.13],'H','Cross-platform direction');d=inp.get('concordance');x=col(d,['bulk_log2fc','bulk_log2_fold_change']);y=col(d,['single_cell_log2fc','single_cell_log2_fold_change']);same=np.sign(d[x])==np.sign(d[y]);a.scatter(d[x],d[y],c=np.where(same,OI['green'],OI['orange']),s=20,edgecolor='white',lw=.3)
 for gene in ['CCL21','CXCL16','CXCR6','MUC2','ICAM1']:
  for _,r in d[d['gene']==gene].iterrows():a.annotate(gene,(r[x],r[y]),xytext=(3,3),textcoords='offset points',fontsize=8)
 a.set_xlabel('Bowel log2 fold change');a.set_ylabel('Single-cell log2 fold change');clean(a);a.legend(handles=[Line2D([],[],marker='o',ls='',color=OI['green'],label='Matching'),Line2D([],[],marker='o',ls='',color=OI['orange'],label='Differing')],loc='upper center',bbox_to_anchor=(.5,-.36),ncol=2,handlelength=.7,handletextpad=.3,columnspacing=.7);inp.save(2,'H',d);finish(fig,2,inp)
def figure3(inp):
 fig=canvas(265);glegend(fig)
 a=panel(fig,[.15,.77,.33,.16],'A','Species abundance and prevalence');d=inp.get('species_effects');pc=col(d,['prevalence_delta_nec_vs_control']);d['display_species']=d['species'].str.replace('s__','',regex=False).str.replace('_',' ',regex=False);den=max(d['mean_abundance_nec_percent'].max(),d['mean_abundance_control_percent'].max());sizes=20+110*np.sqrt(np.maximum(d['mean_abundance_nec_percent'],d['mean_abundance_control_percent'])/den);a.scatter(d['hedges_g'],d[pc],s=sizes,c=np.where(d['hedges_g']>=0,OI['red'],OI['blue']),alpha=.8,edgecolor='black',lw=.3)
 for gene in ['Klebsiella pneumoniae','Clostridium neonatale','Staphylococcus epidermidis']:
  for _,r in d[d['display_species']==gene].iterrows():a.annotate(gene.replace(' ','\n',1),(r['hedges_g'],r[pc]),xytext={'Klebsiella pneumoniae':(12,-24),'Clostridium neonatale':(12,8),'Staphylococcus epidermidis':(4,6)}[gene],textcoords='offset points',fontsize=8)
 a.set_xlabel('Species CLR Hedges g');a.set_ylabel('Prevalence difference');clean(a);inp.save(3,'A',d)
 a=panel(fig,[.64,.77,.32,.16],'B','Infant taxonomic profiles');d=inp.get('species_long');ic=col(d,['infant_id']);vc=col(d,['abundance_percent']);m=d.groupby([ic,'condition','species'],as_index=False)[vc].mean();ordr=m[[ic,'condition']].drop_duplicates().sort_values(['condition',ic])[ic];wide=m.pivot(index=ic,columns='species',values=vc).reindex(ordr).fillna(0);species=wide.mean().nlargest(6).index.tolist();sub=wide[species].copy();sub['Other']=np.maximum(0,100-sub.sum(axis=1));bottom=np.zeros(len(sub));colors=[OI['green'],OI['orange'],OI['purple'],OI['sky'],OI['yellow'],OI['blue'],'white']
 for s,color in zip(sub.columns,colors):a.bar(range(len(sub)),sub[s],bottom=bottom,width=.8,color=color,edgecolor='black' if s=='Other' else 'none',lw=.3);bottom+=sub[s].to_numpy()
 a.set_xticks(range(len(sub)),[str(x) for x in sub.index],rotation=90);a.set_ylabel('Relative abundance (%)');clean(a);labels=[s.replace('s__','').replace('_',' ') for s in sub.columns];fig.legend(handles=[Patch(facecolor=c,edgecolor='black' if s=='Other' else 'none',label=s) for s,c in zip(labels,colors)],loc='upper center',bbox_to_anchor=(.54,.695),ncol=2,columnspacing=1,handlelength=1,handletextpad=.3);inp.save(3,'B',sub.reset_index());inp.save(3,'B_all_taxa',m)
 a=panel(fig,[.28,.385,.20,.18],'C','Microbial pathway profiles');d=inp.get('pathway_effects');lc=col(d,['pathway_id','pathway']);full=d.copy();d=d.sort_values('hedges_g');d=d.iloc[np.unique(np.r_[np.arange(min(6,len(d))),np.arange(max(0,len(d)-6),len(d))])].copy();d['display_pathway']=d[lc].astype(str).str.split(':').str[0];forest(a,d,label='display_pathway',limits=(-3,3));inp.save(3,'C',d);inp.save(3,'C_display_set',full);inp.save(3,'C_complete_family',inp.get('pathway_all_effects'))
 a=panel(fig,[.66,.385,.22,.18],'D','Infant pathway profiles');z=inp.get('pathway_long');ic=col(z,['infant_id']);vc=col(z,['log1p_cpm']);pc=col(z,['pathway','pathway_id']);m=z.groupby([ic,'condition',pc],as_index=False)[vc].mean();mat=m.pivot(index=ic,columns=pc,values=vc);selected=d[col(d,['pathway','pathway_id'])].tolist();mat=mat.reindex(columns=selected);mat=(mat-mat.mean(axis=0))/mat.std(axis=0,ddof=1);mat=mat.reindex(m[[ic,'condition']].drop_duplicates().sort_values(['condition',ic])[ic]);mat.columns=['\n'.join(textwrap.wrap(str(x).split(':')[0],width=12,break_long_words=True,break_on_hyphens=True)) for x in mat.columns];heat(fig,a,mat,-2,2,'Pathway z score');a.set_xticks(range(len(mat.columns)),range(1,len(mat.columns)+1),rotation=90);a.set_xlabel('Pathway order in C');inp.save(3,'D',m)
 a=panel(fig,[.25,.10,.20,.16],'E','Named stool metabolites');d=inp.get('metabolite_effects');lc=col(d,['canonical_metabolite','feature']);d=d.sort_values('hedges_g');d['display_metabolite']=[f'M{i+1} '+str(v) for i,v in enumerate(d[lc])];forest(a,d,label='display_metabolite',xlabel='Infant-level Hedges g');inp.save(3,'E',d);inp.save(3,'E_complete_family',inp.get('metabolite_all_effects'))
 a=panel(fig,[.72,.10,.16,.16],'F','Taxon-metabolite associations');d=inp.get('metabolite_correlations');tax=d.groupby('taxon')['partial_spearman_rho'].apply(lambda x:x.abs().max()).nlargest(7).index.tolist();met=d['metabolite'].drop_duplicates().tolist();md=inp.get('metabolite_effects').sort_values('hedges_g');aliases={'succinic acid':'Succinate','lactic acid':'Lactate','inosine':'Inosine','nicotinic acid':'Nicotinic acid','indole-3-lactic acid':'Indole-3-lactate','indole-3-acetic acid':'Indole-3-acetate','3-hydroxybutyric acid':'3-Hydroxybutyrate','chenodeoxycholic acid':'Chenodeoxycholate','3-hydroxypropionic acid':'3-Hydroxypropionate'};ranks={str(v).casefold():i for i,v in enumerate(md['canonical_metabolite'])};assert all(str(aliases.get(v,v)).casefold() in ranks for v in met);met=sorted(met,key=lambda v:ranks[str(aliases.get(v,v)).casefold()]);d['metabolite_code']=d.metabolite.map({v:f'M{i+1}' for i,v in enumerate(met)});d['taxon_display']=d['taxon'].str.replace('s__','',regex=False).str.replace('_',' ',regex=False);taxdisplay=[x.replace('s__','').replace('_',' ') for x in tax];dots(fig,a,d,'taxon_display','metabolite','partial_spearman_rho','fdr_bh',met,taxdisplay,'Partial Spearman rho',.7);a.set_xticks(range(len(met)),[f'M{i+1}' for i in range(len(met))],rotation=90);a.set_xlabel('Metabolites in E');inp.save(3,'F',d);finish(fig,3,inp)

def figure4(inp):
 fig=canvas(265);glegend(fig);d=inp.get('donor_state_metrics');s=sc(d);ac=col(d,['abundance_fraction','posterior_fraction_all_cells']);pc=col(d,['programme_score','program_score','programme_z']);boxes=[[.15,.79,.33,.15],[.64,.79,.32,.15],[.15,.555,.33,.15]]
 for st,letter,box in zip(['M_cell','LEC','T_reg'],['A','B','C'],boxes):
  a=panel(fig,box,letter,LABEL[st]);t=d[d[s]==st]
  for g,z in t.groupby('condition'):a.scatter(z[ac]*100,z[pc],s=27,c=GROUP[g],edgecolor='white',lw=.4)
  dc=col(t,['donor_id','donor'])
  label_donors(a,t,ac,pc,dc)
  a.set_xlabel('Posterior-expected abundance (%)');a.set_ylabel('Marker score (log2 CPM)');clean(a);inp.save(4,letter,t)
 a=panel(fig,[.73,.555,.23,.15],'D','Adjusted representation-to-\nmarker-score contrast');de=inp.get('contrast_effects');lc=col(de,['label','state','target_state','contrast']);ef=col(de,['coefficient','beta','effect']);lo=col(de,['ci_low','hc3_ci_low']);hi=col(de,['ci_high','hc3_ci_high']);de['display']=de[lc].astype(str).map(lambda x:{'combined':'M + LEC','M_cell':'M cells','LEC':'LECs','T_reg':'Tregs'}.get(x,x));forest(a,de,ef,lo,hi,'display',xlabel='Adjusted diagnosis coefficient');inp.save(4,'D',de)
 a=panel(fig,[.26,.29,.22,.17],'E','Marker estimates and coverage');ge=inp.get('target_gene_effects');mc=col(ge,['method'],False)
 if mc:
  v=ge[mc].astype(str);mask=v.str.contains('posterior',case=False)&~v.str.contains('imput',case=False)
  if mask.any():ge=ge[mask]
 ge=ge[ge['gene'].isin(TGENES)].drop_duplicates(['gene',sc(ge)]).copy();ge['_order']=ge['gene'].map({x:i for i,x in enumerate(TGENES)});ge=ge.sort_values('_order');ef=col(ge,['coefficient','hedges_g_nec_vs_control','hedges_g']);lo=col(ge,['ci_low'],False);hi=col(ge,['ci_high'],False);ge['display_gene']=[str(g)+' ('+str(int(n))+'/11)' for g,n in zip(ge.gene,ge.nonzero_donors)];forest(a,ge,ef,lo or '_none',hi or '_none','display_gene',xlabel='Adjusted standardized effect');inp.save(4,'E',ge)
 a=panel(fig,[.67,.29,.21,.17],'F','Cross-source module effects');mo=inp.get('module_effects');ss=col(mo,['source','accession']);ef=col(mo,['aligned_nec_direction_standardized_effect','effect']);mods=['Antigen sampling','Lymphatic support','Immune residency','Epithelial repair','Inflammatory injury'];mat=mo.pivot_table(index=ss,columns='module',values=ef).reindex(index=['HCA NEC atlas','GSE46619','GSE64801'],columns=mods);mat.columns=[x.replace(' ','\n') for x in mat.columns];heat(fig,a,mat,-3,3,'Aligned effect',90);inp.save(4,'F',mo)
 a=panel(fig,[.15,.075,.33,.12],'G','M-cell maturation projection');ds=inp.get('donor_stage')
 if 'weighting_method' in ds:ds=ds[ds['weighting_method'].astype(str).str.contains('posterior',case=False)]
 if 'reg' in ds:ds=ds[np.isclose(ds['reg'],.05)]
 if 'reg_m' in ds:ds=ds[np.isclose(ds['reg_m'],1.)]
 points(a,ds,'barycentric_maturation_stage','Barycentric maturation stage');a.set_ylim(0,3);a.set_yticks([0,1,2,3]);inp.save(4,'G',ds)
 a=panel(fig,[.65,.075,.23,.12],'H','Donor maturation-stage profiles');st=inp.get('stage_probability');dc=col(st,['donor_id','donor']);stage=col(st,['stage_index','maturation_stage']);mat=st.pivot_table(index=dc,columns=stage,values='transport_probability').reindex(index=order(st));heat(fig,a,mat,0,1,'Transport probability',45,SEQ);a.set_xticks(range(len(mat.columns)),['Stage '+str(i) for i in mat.columns],rotation=45,ha='right');inp.save(4,'H',st);finish(fig,4,inp)

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--figures',nargs='+',type=int,default=[1,2,3,4]);args=p.parse_args();font=font_manager.findfont('Liberation Sans',fallback_to_default=False);inp=Inputs(args.manifest,args.out)
 for n in args.figures:globals()[f'figure{n}'](inp)
 (inp.out/'figure_input_manifest.json').write_text(json.dumps({'font':font,'canvas_width_mm':180,'minimum_font_pt':8,'sources':inp.sources,'cloud_execution':inp.m.get('cloud_execution')},indent=2),encoding='utf-8');print(json.dumps({'completed_figures':args.figures,'output':str(inp.out),'font':font,'sources':len(inp.sources)}))
if __name__=='__main__':main()
