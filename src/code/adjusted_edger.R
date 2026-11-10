suppressPackageStartupMessages({library(edgeR);library(limma)})
r<-normalizePath(commandArgs(trailingOnly=TRUE)[[1]],mustWork=TRUE)
indir<-file.path(r,'04_corrected_results','01_single_cell_corrected');out<-file.path(r,'new_results');dir.create(out,showWarnings=FALSE)
meta<-read.csv(file.path(indir,'pseudobulk_metadata.csv'),stringsAsFactors=FALSE)
clinical<-read.csv(file.path(r,'inputs','donor_metadata.csv'),stringsAsFactors=FALSE)
meta<-merge(meta,clinical[,c('donor_id','gestational_age_weeks','age_at_surgery_days')],by='donor_id',sort=FALSE)
status<-list();targets<-c('SPIB','GP2','PROX1','ACKR4','FOXP3','TIGIT','CXCR6','CCL21','CCL20','IL1B','LYVE1','PDPN','FLT4')
for(state in unique(meta$cell_state)){
 p<-if(state=='all_cells')file.path(indir,'pseudobulk_counts_all_cells.csv.gz') else file.path(indir,'pseudobulk_counts_by_cell_state',paste0('pseudobulk_counts_',state,'.csv.gz'))
 if(!file.exists(p))next
 dat<-read.csv(p,check.names=FALSE);m<-meta[meta$cell_state==state&complete.cases(meta[,c('gestational_age_weeks','age_at_surgery_days')]),];ids<-intersect(names(dat)[-1],m$pseudobulk_id);m<-m[match(ids,m$pseudobulk_id),];counts<-as.matrix(dat[,ids,drop=FALSE]);rownames(counts)<-dat$gene;group<-factor(m$condition,levels=c('control','nec'));design<-model.matrix(~group+scale(gestational_age_weeks)+scale(age_at_surgery_days),data=m)
 groups<-table(group);message(state,' ',paste(groups,collapse=','),' rank=',qr(design)$rank)
 if(any(groups<3)||nrow(design)<=ncol(design)||qr(design)$rank<ncol(design)){status[[state]]<-data.frame(state=state,status='insufficient donors or rank',n_control=groups[['control']],n_nec=groups[['nec']],rank=qr(design)$rank);next}
 y<-DGEList(counts=counts);keep<-filterByExpr(y,design=design);y<-y[keep,,keep.lib.sizes=FALSE];y<-calcNormFactors(y);y<-estimateDisp(y,design,robust=TRUE);fit<-glmQLFit(y,design,robust=TRUE);test<-glmQLFTest(fit,coef=2);tab<-topTags(test,n=Inf)$table;tab$gene<-rownames(tab);tab$state<-state;tab$n_control<-groups[['control']];tab$n_nec<-groups[['nec']];tab$model<-'NEC + gestational age + surgery age';write.csv(tab,file.path(out,paste0('adjusted_edger_',state,'.csv')),row.names=FALSE);write.csv(tab[tab$gene%in%targets,],file.path(out,paste0('adjusted_edger_targets_',state,'.csv')),row.names=FALSE);status[[state]]<-data.frame(state=state,status='completed',n_control=groups[['control']],n_nec=groups[['nec']],rank=qr(design)$rank)
}
write.csv(do.call(rbind,status),file.path(out,'adjusted_edger_status.csv'),row.names=FALSE)
