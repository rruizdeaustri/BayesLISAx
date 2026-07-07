#!/usr/bin/env python3
"""Aggregate static-NS Sangria window scans into campaign CSV tables."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path

def val(r,*names):
    for n in names:
        if n in r and r[n] not in (None,""):
            return r[n]
    return ""
def flt(x):
    try: return float(x)
    except Exception: return float('-inf')
def first(x):
    if isinstance(x,list): return x[0] if x else ""
    return x if x is not None else ""
def dump(x):
    return json.dumps(x) if isinstance(x,(list,dict,tuple)) else ("" if x is None else x)

p=argparse.ArgumentParser()
p.add_argument('--inputs',nargs='+',required=True)
p.add_argument('--out-all',required=True)
p.add_argument('--out-k-summary',required=True)
p.add_argument('--out-selected',required=True)
p.add_argument('--highres-inputs',nargs='*',default=[])
p.add_argument('--out-highres',default='')
a=p.parse_args()
rows=[]
for path in a.inputs:
    r=json.loads(Path(path).read_text())
    meta=r.get('metadata',{})
    rows.append({
        'window_id': val(meta,'window_id','band_id'), 'f_min': val(meta,'f_min'), 'f_max': val(meta,'f_max'),
        'K': int(val(meta,'K','kmax') or 0), 'seed': r.get('seed',''), 'logZ': r.get('logZ',''),
        'logZ_err': r.get('logZ_std',''), 'best_logL': r.get('best_logL',''), 'ESS': r.get('ESS',''),
        'best_f0': dump(first(r.get('f0_best'))), 'posterior_f0_mean': dump(first(r.get('f0_mean'))),
        'posterior_f0_std': dump(first(r.get('f0_std'))), 'all_best_f0': dump(r.get('f0_best')),
        'all_posterior_f0_mean': dump(r.get('f0_mean')), 'all_posterior_f0_std': dump(r.get('f0_std')),
        'runtime_seconds': r.get('runtime_seconds',''), 'status': r.get('status',''), 'return_code': r.get('return_code',''),
        'summary_json': path, 'log_path': r.get('log_path','')})
rows.sort(key=lambda r:(str(r['window_id']),int(r['K']),int(r['seed'] or 0)))
fields=list(rows[0].keys()) if rows else ['window_id','f_min','f_max','K','seed','logZ','logZ_err','best_logL','ESS','best_f0','posterior_f0_mean','posterior_f0_std','runtime_seconds']
for out in [a.out_all]:
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    with open(out,'w',newline='') as f: w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)

ks=[]; selected=[]
for wid in sorted({r['window_id'] for r in rows}):
    wr=[r for r in rows if r['window_id']==wid]
    byk={}
    for r in wr: byk.setdefault(r['K'],[]).append(r)
    prev=None
    summaries=[]
    for k in sorted(byk):
        krows=byk[k]
        br=max(krows, key=lambda r: flt(r['logZ']))
        logzs=[flt(r['logZ']) for r in krows if math.isfinite(flt(r['logZ']))]
        mean_logz=sum(logzs)/len(logzs) if logzs else ''
        std_logz=(sum((x-mean_logz)**2 for x in logzs)/(len(logzs)-1))**0.5 if len(logzs)>1 else (0.0 if len(logzs)==1 else '')
        best_logz=flt(br['logZ'])
        dz='' if prev is None else best_logz-prev
        summaries.append({
            'window_id':wid,'f_min':br['f_min'],'f_max':br['f_max'],'K':k,
            'n_seeds':len(krows),'best_logZ':br['logZ'],'best_logZ_seed':br['seed'],
            'mean_logZ':mean_logz,'std_logZ':std_logz,
            'min_logZ':min(logzs) if logzs else '','max_logZ':max(logzs) if logzs else '',
            'max_logZ_err':br['logZ_err'],'delta_logZ_from_previous_K':dz})
        prev=best_logz
    ks.extend(summaries)
    best=max(summaries, key=lambda r: flt(r['best_logZ']))
    selected.append({'window_id':wid,'f_min':best['f_min'],'f_max':best['f_max'],'K_best':best['K'],'best_seed':best['best_logZ_seed'],'max_logZ':best['best_logZ'],'max_logZ_err':best['max_logZ_err']})
for out,data,fields2 in [(a.out_k_summary,ks,['window_id','f_min','f_max','K','n_seeds','best_logZ','best_logZ_seed','mean_logZ','std_logZ','min_logZ','max_logZ','max_logZ_err','delta_logZ_from_previous_K']),(a.out_selected,selected,['window_id','f_min','f_max','K_best','best_seed','max_logZ','max_logZ_err'])]:
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    with open(out,'w',newline='') as f: w=csv.DictWriter(f,fields2); w.writeheader(); w.writerows(data)

hi=[]
for path in a.highres_inputs:
    r=json.loads(Path(path).read_text()); meta=r.get('metadata',{})
    hi.append({'window_id':val(meta,'window_id','band_id'),'f_min':val(meta,'f_min'),'f_max':val(meta,'f_max'),'K':val(meta,'K','kmax'),'seed':r.get('seed',''),'logZ':r.get('logZ',''),'logZ_err':r.get('logZ_std',''),'best_logL':r.get('best_logL',''),'ESS':r.get('ESS',''),'best_f0':dump(first(r.get('f0_best'))),'posterior_f0_mean':dump(first(r.get('f0_mean'))),'posterior_f0_std':dump(first(r.get('f0_std'))),'runtime_seconds':r.get('runtime_seconds',''),'status':r.get('status',''),'summary_json':path,'log_path':r.get('log_path','')})
fields3=list(hi[0].keys()) if hi else ['window_id','f_min','f_max','K','seed','logZ','logZ_err','best_logL','ESS','best_f0','posterior_f0_mean','posterior_f0_std','runtime_seconds','status']
if a.out_highres:
    Path(a.out_highres).parent.mkdir(parents=True,exist_ok=True)
    with open(a.out_highres,'w',newline='') as f: w=csv.DictWriter(f,fields3); w.writeheader(); w.writerows(hi)
