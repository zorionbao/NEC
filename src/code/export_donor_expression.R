suppressPackageStartupMessages({library(edgeR);library(limma)})
r<-normalizePath(commandArgs(trailingOnly=TRUE)[[1]],mustWork=TRUE);indir<-file.path(r,'04_corrected_results','01_single_cell_corrected');out<-file.path(r,'new_results');meta<-read.csv(file.path(indir,'pseudobulk_metadata.csv'));clinical<-read.csv(file.path(r,'inputs','donor_metadata.csv'));meta<-merge(meta,clinical[,c('donor_id','gestational_age_weeks','age_at_surgery_days')],by='donor_id',sort=FALSE)
genes<-c('HCAR2','CSF2','CXCL1','VCAM1','SELE','CSF3','TNIP3','TNFAIP6','CCL3','CCL21','IL1B','CCL20','IFITM3','FOXC2','GP2','SPIB','PROX1','ACKR4','FOXP3','TIGIT','CXCR6')
prepare<-function(state,omit=NULL){
 p<-if(state=='all_cells')file.path(indir,'pseudobulk_counts_all_cells.csv.gz') else file.path(indir,'pseudobulk_counts_by_cell_state',paste0('pseudobulk_counts_',state,'.csv.gz'));dat<-read.csv(p,check.names=FALSE);m<-meta[meta$cell_state==state&!(meta$donor_id%in%omit),];ids<-intersect(names(dat)[-1],m$pseudobulk_id);m<-m[match(ids,m$pseudobulk_id),];counts<-as.matrix(dat[,ids,drop=FALSE]);rownames(counts)<-dat$gene;group<-factor(m$condition,levels=c('control','nec'));design<-model.matrix(~group+scale(gestational_age_weeks)+scale(age_at_surgery_days),m);y<-DGEList(counts=counts);keep<-filterByExpr(y,design=design);y<-y[keep,,keep.lib.sizes=FALSE];y<-calcNormFactors(y);list(y=y,m=m,design=design,counts=counts)
}
rows<-list()
for(state in c('all_cells','native_vascular_endothelial','vascular_endothelial','native_LEC')){
 a<-prepare(state);cpmat<-cpm(a$y,log=TRUE,prior.count=2)
 for(gene in intersect(genes,rownames(a$counts)))for(j in seq_len(nrow(a$m))){
  m<-a$m[j,];rows[[length(rows)+1]]<-data.frame(donor_id=m$donor_id,condition=m$condition,state=state,gene=gene,n_cells=m$n_cells,raw_count=a$counts[gene,j],TMM_log2_CPM=if(gene%in%rownames(cpmat))cpmat[gene,j] else NA_real_,filtered_library_size=a$y$samples$lib.size[j],TMM_norm_factor=a$y$samples$norm.factors[j],expression_note='edgeR TMM log2 CPM, prior.count=2; same diagnosis+GA+surgery design count filter')
 }
}
write.csv(do.call(rbind,rows),file.path(out,'normalized_donor_expression_selected.csv'),row.names=FALSE)
rows<-list()
for(donor in unique(meta$donor_id[meta$cell_state=='all_cells'])){
 a<-prepare('all_cells',donor);a$y<-estimateDisp(a$y,a$design,robust=TRUE);fit<-glmQLFit(a$y,a$design,robust=TRUE);test<-glmQLFTest(fit,coef=2);tab<-topTags(test,n=Inf)$table
 if('HCAR2'%in%rownames(tab)){row<-tab['HCAR2',,drop=FALSE];row$gene<-'HCAR2';row$omitted_donor<-donor;row$n_control<-sum(a$m$condition=='control');row$n_nec<-sum(a$m$condition=='nec');rows[[length(rows)+1]]<-row}
}
write.csv(do.call(rbind,rows),file.path(out,'HCAR2_leave_one_donor_adjusted_edger.csv'),row.names=FALSE);print(do.call(rbind,rows))
