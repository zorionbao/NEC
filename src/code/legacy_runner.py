from pathlib import Path
import importlib,sys,pandas as pd
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'legacy_analysis/code'));common=importlib.import_module('40_story_chain_common');module=sys.argv[1];m=importlib.import_module(module)
if module.startswith('42_'):
 def collect():
  posterior=pd.read_csv(r/'inputs/native_posteriors.csv.gz',dtype={'donor_id':str});m.save_csv_gzip(posterior,'uplift_full_scanvi_posteriors.csv.gz');return posterior
 m.collect_full_posteriors=collect
m.main()
