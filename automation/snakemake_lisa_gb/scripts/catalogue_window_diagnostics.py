#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,ast
from pathlib import Path
import numpy as np
from sangria_catalog import read_catalogues, concatenate_catalogues, lisa_gb_approx_snr

def read_selected(path):
    with open(path,newline='') as f: return list(csv.DictReader(f))
def parse_f0(x):
    if not x: return []
    try: v=ast.literal_eval(x)
    except Exception:
        try: return [float(x)]
        except Exception: return []
    if isinstance(v,(list,tuple)): return [float(y) for y in v]
    return [float(v)]

p=argparse.ArgumentParser()
p.add_argument('--h5-path',required=True)
p.add_argument('--selected-csv',required=True)
p.add_argument('--highres-csv',default='')
p.add_argument('--tobs',type=float,required=True)
p.add_argument('--margin-hz',type=float,required=True)
p.add_argument('--snr-thresholds',nargs='+',type=float,required=True)
p.add_argument('--top-n',type=int,default=40)
p.add_argument('--out-summary',required=True)
p.add_argument('--out-matches',required=True)
a=p.parse_args()
cat=concatenate_catalogues(read_catalogues(a.h5_path))
f=cat['Frequency']; snr=lisa_gb_approx_snr(f,cat['Amplitude'],a.tobs,cat.get('EclipticLatitude'),cat.get('Inclination'))
sel=read_selected(a.selected_csv)
recovered={}
if a.highres_csv and Path(a.highres_csv).exists():
    with open(a.highres_csv,newline='') as fh:
        for r in csv.DictReader(fh): recovered.setdefault(r['window_id'],[]).extend(parse_f0(r.get('all_best_f0') or r.get('best_f0','')))
summary=[]; matches=[]
for w in sel:
    wid=w['window_id']; lo=float(w['f_min'])-a.margin_hz; hi=float(w['f_max'])+a.margin_hz
    m=(f>=lo)&(f<=hi); idx=np.flatnonzero(m); order=idx[np.argsort(snr[idx])[::-1]] if idx.size else np.array([],dtype=int)
    row={'window_id':wid,'f_min':w['f_min'],'f_max':w['f_max'],'catalogue_f_min':lo,'catalogue_f_max':hi,'total_catalogue_sources':int(idx.size),'top_sources_json':json.dumps([{'rank':i+1,'frequency':float(f[j]),'snr':float(snr[j]),'source_type':str(cat['source_type'][j])} for i,j in enumerate(order[:a.top_n])])}
    for th in a.snr_thresholds: row[f'count_snr_gt_{th:g}']=int(np.sum(snr[idx]>th))
    summary.append(row)
    for rf0 in recovered.get(wid,[]):
        if idx.size:
            j=idx[np.argmin(np.abs(f[idx]-rf0))]
            matches.append({'window_id':wid,'recovered_f0':rf0,'catalogue_frequency':float(f[j]),'delta_f':float(rf0-f[j]),'approx_snr':float(snr[j]),'source_type':str(cat['source_type'][j])})
fields=['window_id','f_min','f_max','catalogue_f_min','catalogue_f_max','total_catalogue_sources']+[f'count_snr_gt_{th:g}' for th in a.snr_thresholds]+['top_sources_json']
Path(a.out_summary).parent.mkdir(parents=True,exist_ok=True)
with open(a.out_summary,'w',newline='') as f1: w=csv.DictWriter(f1,fields); w.writeheader(); w.writerows(summary)
with open(a.out_matches,'w',newline='') as f2: w=csv.DictWriter(f2,['window_id','recovered_f0','catalogue_frequency','delta_f','approx_snr','source_type']); w.writeheader(); w.writerows(matches)
