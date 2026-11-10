#!/usr/bin/env python3
"""Run the research scripts in a data workspace outside this code repository."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parent
SOURCES = REPOSITORY / 'src'


def plan(workspace: Path, python: str, native_python: str, rscript: str):
    w = str(workspace)
    sc = workspace / '04_corrected_results/01_single_cell_corrected'
    st = workspace / '04_corrected_results/02_statistical_reanalysis'
    nr = workspace / 'new_results'
    story = workspace / 'legacy_analysis/results/story_chain_upgrade'
    derived = workspace / 'publication_figures/derived'
    rendered = workspace / 'publication_figures/rendered'
    def py(name, *args):
        return [python, str(workspace / 'code' / name), *map(str, args)]
    def legacy(name, *args):
        return [python, str(workspace / 'legacy_analysis/code' / name), *map(str, args)]
    def plot(name, *args):
        return [python, str(workspace / 'plot_code' / name), *map(str, args)]
    def r(name, *args):
        return [rscript, str(workspace / name), *map(str, args)]
    return {
        'raw-atlas': [py('03_rerun_single_cell_atlas.py', '--raw-root', workspace/'data/raw/HCA_NEC/Cell_ranger_output', '--snapshot-results', workspace/'04_corrected_results/00_remote_snapshot/results', '--output-dir', sc)],
        'reference-download': [legacy('08_download_drcmap_references.py')],
        'reference-prepare': [[native_python, str(workspace/'legacy_analysis/code/09_prepare_drcmap.py')]],
        'reference-train': [[native_python, str(workspace/'legacy_analysis/code/10_run_drcmap.py')]],
        'native-infer': [[native_python, str(workspace/'code/infer_saved_scanvi.py')]],
        'primary': [py('analyze_revision.py', '--root', w), py('gene7_adjusted.py', '--new-results', nr), py('primary_model_diagnostics.py', '--new-results', nr), py('programme_sensitivity.py', '--new-results', nr), py('confident_cell_sensitivity.py'), py('weighted_endothelial_sensitivity.py')],
        'pseudobulk': [py('export_pseudobulk.py'), py('export_annotation_sensitivity.py', '--native')],
        'unadjusted-de': [r('code/04_recompute_count_based_de.R', w)],
        'adjusted-de': [r('code/adjusted_edger.R', w), r('code/export_donor_expression.R', w)],
        'host-statistics': [py('05_single_cell_tissue_mechanism_statistics.py', '--revision-root', w), py('export_display_logcpm.py', '--atlas', sc/'corrected_hca_nec_atlas.h5ad', '--out', nr)],
        'microbial-statistics': [py('06_microbiome_metabolome_statistics.py', '--revision-root', w)],
        'classification': [py('07_cross_sectional_endotype_model.py', '--revision-root', w)],
        'pathway-coverage': [legacy('01_scope_nec.py')],
        'longitudinal': [legacy('05_tempo_nec.py')],
        'bulk-deconvolution': [legacy('18_coder_nec_deconvolution.py')],
        'bulk-adjustment': [r('legacy_analysis/code/43_state_resolve.R')],
        'specificity': [py('legacy_runner.py', '41_specific_nec')],
        'posterior-assignment': [[native_python, str(workspace/'code/legacy_runner.py'), '42_uplift_nec']],
        'cross-source': [py('legacy_runner.py', '44_bridge_nec')],
        'organoid-trends': [py('prepare_organoid_trends.py')],
        'transport': [py('legacy_runner.py', '45_gate_ot')],
        'methylation-download': [legacy('14_download_eclipse_nec.py')],
        'methylation-bins': [legacy('15_prepare_eclipse_bins.py')],
        'methylation-models': [py('rerun_eclipse_full.py', '--root', w)],
        'main-figures': [plot('prepare_figure_inputs.py', '--stats-root', st, '--single-cell', sc, '--new-results', nr, '--story-root', story, '--out', derived), plot('render_revision_figures.py', '--manifest', derived/'figure_inputs.json', '--out', rendered)],
        'atlas-diagnostics': [plot('prepare_supplementary1.py', '--raw-root', sc, '--stats-root', st, '--donors', nr/'donor_covariates.csv', '--out', derived/'supplementary1'), plot('render_supplementary1.py', '--raw-root', sc, '--reviewer-tables', derived/'supplementary1', '--out', rendered)],
        'supplementary-figures': [plot('render_supplementary2.py', '--new-results', nr, '--story-root', story, '--out', rendered), plot('render_supplementary3.py', '--results', w, '--output', rendered), plot('render_supplementary8.py', '--uplift-root', story/'uplift_nec', '--out', rendered), plot('render_supplementary10.py')],
    }


# Prerequisites are files or directories in an external analysis workspace.
# The scripts also validate their internal columns and sample matching.
REQUIRED = {
    'raw-atlas':['data/raw/HCA_NEC/Cell_ranger_output', '04_corrected_results/00_remote_snapshot/results'],
    'reference-prepare':['legacy_analysis/data/public/dr_cmap', '04_corrected_results/01_single_cell_corrected/corrected_hca_nec_atlas.h5ad'],
    'reference-train':['legacy_analysis/data/processed/dr_cmap'],
    'native-infer':['legacy_analysis/results/dr_cmap/models', 'legacy_analysis/results/dr_cmap/epithelial_drcmap_mapped.h5ad'],
    'primary':['inputs/native_posteriors.csv.gz', 'inputs/donor_metadata.csv', 'inputs/liana_native_rank_aggregate.csv', 'legacy_analysis/data/processed/dr_cmap', '04_corrected_results/01_single_cell_corrected/corrected_hca_nec_atlas.h5ad'],
    'pseudobulk':['inputs/native_posteriors.csv.gz', '04_corrected_results/01_single_cell_corrected/corrected_hca_nec_atlas.h5ad'],
    'unadjusted-de':['04_corrected_results/01_single_cell_corrected/pseudobulk_metadata.csv', '04_corrected_results/00_remote_snapshot/results/bulk'],
    'adjusted-de':['04_corrected_results/01_single_cell_corrected/pseudobulk_metadata.csv', 'inputs/donor_metadata.csv'],
    'host-statistics':['04_corrected_results/01_single_cell_corrected/corrected_hca_nec_atlas.h5ad', '04_corrected_results/00_remote_snapshot/results'],
    'microbial-statistics':['04_corrected_results/00_remote_snapshot/results/microbiome_strict', '04_corrected_results/00_remote_snapshot/results/microbiome_metabolome'],
    'classification':['04_corrected_results/02_statistical_reanalysis'],
    'pathway-coverage':['04_corrected_results/00_remote_snapshot/results/microbiome_strict/humann_native_subset/samples'],
    'longitudinal':['legacy_analysis/data/public/preterm_infant_analysis/.RData'],
    'bulk-deconvolution':['04_corrected_results/00_remote_snapshot/results/bulk', '04_corrected_results/01_single_cell_corrected/corrected_hca_nec_atlas.h5ad'],
    'bulk-adjustment':['legacy_analysis/results/coder_nec', '04_corrected_results/02_statistical_reanalysis'],
    'specificity':['04_corrected_results/01_single_cell_corrected/corrected_hca_nec_atlas.h5ad', '04_corrected_results/02_statistical_reanalysis'],
    'posterior-assignment':['inputs/native_posteriors.csv.gz', '04_corrected_results/01_single_cell_corrected/corrected_hca_nec_atlas.h5ad'],
    'cross-source':['legacy_analysis/results/story_chain_upgrade/uplift_nec', 'legacy_analysis/results/coder_nec', '04_corrected_results/02_statistical_reanalysis'],
    'organoid-trends':['legacy_analysis/data/public/dr_cmap/GSE275771_P11N_M_cell_bulk_seq_NormalizedCounts.csv.gz'],
    'transport':['legacy_analysis/results/story_chain_upgrade/uplift_nec', 'legacy_analysis/results/fxr_ferroptosis_ilc3/organoid_mcell_gene_trends.parquet'],
    'methylation-bins':['legacy_analysis/data/public/eclipse_nec/eclipse_sample_metadata.csv', 'legacy_analysis/data/public/eclipse_nec/raw_cov'],
    'methylation-models':['legacy_analysis/data/processed/eclipse_nec/bin_1kb', 'legacy_analysis/data/public/eclipse_nec/eclipse_sample_metadata.csv', '04_corrected_results/02_statistical_reanalysis'],
    'main-figures':['new_results/target_validation.csv', 'new_results/normalized_donor_expression_selected.csv', '04_corrected_results/02_statistical_reanalysis', 'legacy_analysis/results/story_chain_upgrade'],
    'atlas-diagnostics':['04_corrected_results/01_single_cell_corrected/corrected_hca_nec_atlas.h5ad', '04_corrected_results/01_single_cell_corrected/single_cell_rebuild_status.json', '04_corrected_results/01_single_cell_corrected/cell_state_proportions_capture_unit.csv', '04_corrected_results/02_statistical_reanalysis/single_cell_module_scores_capture_unit.csv', '04_corrected_results/02_statistical_reanalysis/single_cell_host_state_map.csv', '04_corrected_results/02_statistical_reanalysis/single_cell_donor_composition_clr_wide.csv', 'new_results/donor_covariates.csv'],
    'supplementary-figures':['new_results', 'legacy_analysis/results/story_chain_upgrade/uplift_nec', 'legacy_analysis/results/eclipse_nec'],
}


def install_sources(workspace: Path):
    records = []
    for source in sorted(SOURCES.rglob('*')):
        if not source.is_file():
            continue
        relative = source.relative_to(SOURCES)
        target = workspace / relative
        if target.is_file() and target.read_bytes() != source.read_bytes():
            raise RuntimeError(f'Existing different script is protected: {target}. Use a fresh workspace.')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        records.append({'file':relative.as_posix(), 'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    for name in ['logs', 'inputs', 'new_results']:
        (workspace/name).mkdir(parents=True, exist_ok=True)
    (workspace/'logs/installed_code_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='list', help='Stage name, list, or install')
    parser.add_argument('--workspace', type=Path, help='External directory containing public inputs and generated outputs')
    parser.add_argument('--python', default=sys.executable, help='Statistical Python interpreter (raw-atlas: use its dedicated interpreter)')
    parser.add_argument('--native-python', default=sys.executable, help='scVI/scANVI Python interpreter')
    parser.add_argument('--rscript', default='Rscript')
    parser.add_argument('--check', action='store_true', help='Check prerequisites and print commands; write nothing')
    args = parser.parse_args()
    stages = plan(Path('.'), args.python, args.native_python, args.rscript)
    if args.stage == 'list':
        print('\n'.join(stages))
        return
    if not args.workspace:
        parser.error('--workspace is required for this stage')
    workspace = args.workspace.expanduser().resolve()
    if workspace == REPOSITORY or REPOSITORY in workspace.parents:
        parser.error('Use an external workspace so research data stay outside the code repository')
    if args.stage not in stages and args.stage != 'install':
        parser.error('Unknown stage; use list')
    missing = [p for p in REQUIRED.get(args.stage,[]) if not (workspace/p).exists()]
    commands = plan(workspace,args.python,args.native_python,args.rscript).get(args.stage,[])
    if args.check:
        print(json.dumps({'stage':args.stage,'missing':missing,'commands':commands},indent=2))
        raise SystemExit(2 if missing else 0)
    if missing:
        parser.error('Missing prerequisites: '+', '.join(missing)+'; see docs/input_contracts.md')
    install_sources(workspace)
    environment = os.environ.copy()
    environment.update({'NEC_RSCRIPT':args.rscript,'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8'})
    for index,command in enumerate(commands,1):
        log = workspace/'logs'/f'{args.stage}_{index}.log'
        print(shlex.join(command),flush=True)
        with log.open('w',encoding='utf-8') as output:
            subprocess.run(command,cwd=workspace,env=environment,stdout=output,stderr=subprocess.STDOUT,check=True)
    print(f'Completed: {args.stage}')


if __name__ == '__main__':
    main()
