"""Pinned optional PP-DocLayout-S: real CPU parity and fresh extraction evidence."""
import argparse
import hashlib
import zipfile
import json
import math
from pathlib import Path
import statistics
import subprocess
import tempfile
import tomllib

import financial_benchmark as f
import layout_benchmark as old
from download_clip import digest,download

ROOT,CRATE=f.ROOT,f.CRATE
DATA=CRATE/'models/doc-layout-public'
MODEL=CRATE/'models/pp-doclayout-s'
SPEC=CRATE/'model-manifests/doc-layout-model.json'
MANIFEST=CRATE/'model-manifests/doc-layout-samples.json'
PROFILE=CRATE/'model-manifests/doc-layout-profile.json'
HISTORY=CRATE/'model-manifests/doc-layout-history.json'
REPORT=CRATE/'docs/doc-layout-results.json'
PARITY=CRATE/'docs/doc-layout-parity.json'
REFERENCE=MODEL/'reference-env/bin/python'
DOMAINS=('receipt','invoice','cheque')
SOURCES=f.SOURCES+[CRATE/'src/lib.rs',CRATE/'src/documents/regions.rs',CRATE/'examples/doc_layout_batch.rs']+sorted((CRATE/'src/document_layout').glob('*.rs'))


def sources():return {str(p.relative_to(ROOT)):digest(p) for p in SOURCES}


def verify_history():
    for name,sha in f.read(HISTORY).items():
        if digest(CRATE/name)!=sha:raise ValueError('historical artifact changed')


def verify_data():
    m=f.read(MANIFEST)
    old.validate(m,[f.read(f.MANIFEST),f.read(CRATE/'model-manifests/ocr-model-samples.json'),f.read(old.MANIFEST)])
    for s in m['samples']:
        for a in s['artifacts']:
            p=DATA/a['path']
            if p.stat().st_size!=a['size'] or digest(p)!=a['sha256']:raise ValueError('document checksum mismatch')
        if s['source'] in ('sroie','zenodo'):
            labels=f.read(DATA/(s['id']+('.json' if s['source']=='sroie' else '.txt')))
            pairs=[('merchant','company'),('date','date'),('total','total')]
            if s['source']=='zenodo':pairs += [('tax','iva_amount'),('reference_number','invoice_number')]
            if s['expected']!={k:labels[v] for k,v in pairs if labels[v]}:raise ValueError('published labels changed')
    verify_history()


def verify_artifacts():
    dependencies=tomllib.loads((ROOT/'Cargo.toml').read_text())['workspace']['dependencies']
    if dependencies['rusto']['version']!='=0.3.2' or dependencies['rten']['version']!='=0.27.0':
        raise ValueError('pinned RustO/RTen versions changed')
    spec=f.read(SPEC)
    for a in spec['artifacts']:
        p=MODEL/a['name']
        if p.stat().st_size!=a['size'] or digest(p)!=a['sha256']:raise ValueError('model artifact checksum mismatch')
    if digest(MODEL/'pp_doclayout_s.onnx')!=spec['prepared_sha256']:raise ValueError('prepared ONNX checksum mismatch')
    labels=(MODEL/'upstream-inference.yml').read_text().split('label_list:\n')[1].split('Hpi:')[0]
    if [l[2:] for l in labels.splitlines() if l.startswith('- ')]!=spec['classes'] or len(spec['classes'])!=23:
        raise ValueError('official class mapping changed')


def native(jobs,output):
    cmd=['rtk','proxy','mise','exec','--','cargo','run','-p','rs-ocr','--features','document-layout','--release','--locked','--example','doc_layout_batch','--',str(MODEL/'pp_doclayout_s.onnx'),str(CRATE/'models/ppocrv6-tiny')]
    result=subprocess.run(cmd,cwd=ROOT,input=''.join(json.dumps(j)+'\n' for j in jobs),text=True,stdout=subprocess.PIPE,check=True)
    rows=[json.loads(l) for l in result.stdout.splitlines()]
    if {r['id'] for r in rows}!={j['id'] for j in jobs} or len(rows)!=len(jobs) or any('error' in r for r in rows):
        raise ValueError('native inference incomplete or failed: '+str([r.get('error') for r in rows if 'error' in r]))
    output.write_text(result.stdout)
    return {r['id']:r for r in rows}


def parity_jobs():
    manifest=f.read(MANIFEST);cached=f.predictions(old.DATA/'baseline.jsonl');jobs=[]
    for id in f.read(SPEC)['parity_samples']:
        s=next(s for s in manifest['samples'] if s['id']==id)
        jobs.append(dict(cached[id],image=str(DATA/s['image']),decimal_separator=s['decimal_separator']))
    return jobs


def validate_reference_regions(regions):
    for region in regions:
        code=region['class_id'];score=region['confidence'];box=region['raw_bounds']
        if type(code) is not int or code not in range(23) or len(box)!=4:
            raise ValueError('invalid reference class/box shape')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in [score,*box]) \
                or not 0<=score<=1 or box[2]<=box[0] or box[3]<=box[1]:
            raise ValueError('non-finite or invalid reference score/geometry')


def compare_parity(native_rows,reference):
    tolerances=f.read(SPEC)['parity_tolerances'];comparisons=[];positive=0;negative=0
    ids=[r['id'] for r in reference['predictions']]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate reference IDs')
    for ref in reference['predictions']:
        row=native_rows[ref['id']]['layout'];actual=list(row['regions']);expected=ref['regions']
        validate_reference_regions(actual);validate_reference_regions(expected)
        labels=f.read(SPEC)['classes']
        if any(o['class_name']!=labels[o['class_id']] for o in actual):raise ValueError('official class name mismatch')
        if (row['width'],row['height'])!=(ref['width'],ref['height']) or len(actual)!=len(expected):
            raise ValueError('reference dimensions or region count mismatch')
        positive+=bool(expected);negative+=not expected
        max_score=max_coordinates=0.0
        for exp in expected:
            candidates=[a for a in actual if a['class_id']==exp['class_id']]
            if not candidates:raise ValueError('class mismatch')
            got=min(candidates,key=lambda a:max(abs(x-y) for x,y in zip(a['raw_bounds'],exp['raw_bounds'])))
            score=abs(got['confidence']-exp['confidence']);coordinate=max(abs(x-y) for x,y in zip(got['raw_bounds'],exp['raw_bounds']))
            if score>tolerances['max_score_error'] or coordinate>tolerances['max_coordinate_error_pixels']:
                raise ValueError('reference score/coordinate tolerance exceeded')
            actual.remove(got);max_score=max(max_score,score);max_coordinates=max(max_coordinates,coordinate)
        comparisons.append({'id':ref['id'],'width':ref['width'],'height':ref['height'],'regions':len(expected),'max_score_error':max_score,'max_coordinate_error_pixels':max_coordinates})
    if set(native_rows)!={r['id'] for r in reference['predictions']} or positive<3 or negative<1:
        raise ValueError('parity requires nonempty positive and empty negative controls')
    return comparisons


def model_parity(write=False):
    verify_data();verify_artifacts();jobs=parity_jobs()
    with tempfile.TemporaryDirectory() as tmp:
        directory=Path(tmp);jobfile=directory/'jobs.json';f.write(jobfile,jobs)
        rows=native(jobs,directory/'native.jsonl')
        output=directory/'reference.json'
        subprocess.run(['rtk','proxy','mise','exec','--',str(REFERENCE),str(CRATE/'scripts/doc_layout_reference.py'),'reference','--jobs',str(jobfile),'--output',str(output)],cwd=ROOT,check=True)
        reference=f.read(output);comparisons=compare_parity(rows,reference)
        result={'version':1,'spec_sha256':digest(SPEC),'reference_script_sha256':digest(CRATE/'scripts/doc_layout_reference.py'),
            'reference_runtime':reference['runtime'],'comparisons':comparisons,'tolerances':f.read(SPEC)['parity_tolerances'],
            'limits':'Four specified public images only; independent OpenCV cubic preprocessing plus ONNX Runtime CPU. Native Rust cubic RGB rounding may differ by one unit. Does not establish Paddle native parity or every input branch.'}
        # Floating-point implementation details can vary across CPUs; retain
        # measured original errors, verify current runs against the same limits.
        if write:
            f.write(PARITY,result);f.write(MODEL/'reference.json',reference)
            (MODEL/'parity-native.jsonl').write_text((directory/'native.jsonl').read_text())
        else:
            recorded=f.read(PARITY)
            if recorded['spec_sha256']!=result['spec_sha256'] or recorded['reference_script_sha256']!=result['reference_script_sha256'] or recorded['reference_runtime']!=result['reference_runtime']:
                raise ValueError('stale parity source binding')
            compare_parity(f.predictions(MODEL/'parity-native.jsonl'),f.read(MODEL/'reference.json'))
    print('DOC_LAYOUT_MODEL_VERIFIED')


def views(rows,version):return {id:dict(r,fields=r[version]['fields'],issues=r[version]['issues']) for id,r in rows.items()}


def check_results(samples,rows):
    for version in ('baseline','assisted'):old.check_rows(samples,views(rows,version))
    for row in rows.values():
        regions=row['layout']['regions'];members=row['layout']['region_spans']
        if len(regions)!=len(members):raise ValueError('region membership mismatch')
        for name,field in row['baseline']['fields'].items():
            after=row['assisted']['fields'][name]
            if field['candidates'] and after!=field:raise ValueError('existing page candidates were overwritten')
            if not field['candidates']:
                for c in after['candidates']:
                    if not any(set(c['span_indices'])<=set(m) for m in members):raise ValueError('backfill lacks containing region evidence')
                    if 'PP-DocLayout-S region association' not in row['assisted']['issues']:raise ValueError('unreviewed model backfill')


def run(split):
    verify_data();verify_artifacts();manifest=f.read(MANIFEST)
    if PROFILE.exists():
        profile=f.read(PROFILE)
        if profile['sources']!=sources() or profile['manifest_sha256']!=digest(MANIFEST) or profile['spec_sha256']!=digest(SPEC):
            raise ValueError('implementation/data changed after freeze')
    output=DATA/f'{split}.jsonl'
    if split=='test' and output.exists():raise ValueError('refusing to overwrite held-out output')
    cached=f.predictions(old.DATA/'baseline.jsonl');samples=[s for s in manifest['samples'] if s['split']==split];jobs=[]
    for s in samples:
        job={'id':s['id'],'kind':s['kind'],'image':str(DATA/s['image']),'decimal_separator':s['decimal_separator']}
        if split=='dev':job['spans']=cached[s['id']]['spans']
        jobs.append(job)
    rows=native(jobs,output);check_results(samples,rows)
    print('DOC_LAYOUT_BATCH_COMPLETED',split,len(rows))


def freeze():
    if (DATA/'test.jsonl').exists():raise ValueError('cannot refreeze after held-out results')
    verify_data();verify_artifacts();m=f.read(MANIFEST)
    check_results([s for s in m['samples'] if s['split']=='dev'],f.predictions(DATA/'dev.jsonl'))
    f.write(PROFILE,{'sources':sources(),'manifest_sha256':digest(MANIFEST),'spec_sha256':digest(SPEC),
        'development_sha256':digest(DATA/'dev.jsonl'),'history_sha256':digest(HISTORY),'parity_sha256':digest(PARITY),
        'tiny_models':f.read(f.PROFILE)['models']})
    print('DOC_LAYOUT_FROZEN')


def report():
    verify_data();verify_artifacts();profile=f.read(PROFILE)
    f.verify_sources(profile['sources'],archive=DATA/'frozen-sources')
    if profile['manifest_sha256']!=digest(MANIFEST) or profile['spec_sha256']!=digest(SPEC):raise ValueError('frozen data changed')
    if profile['development_sha256']!=digest(DATA/'dev.jsonl') or profile['history_sha256']!=digest(HISTORY) or profile['parity_sha256']!=digest(PARITY):raise ValueError('frozen evidence changed')
    if any(digest(ROOT/p)!=sha for p,sha in profile['tiny_models'].items()):raise ValueError('Tiny weights changed')
    m=f.read(MANIFEST);metrics={};failures=[];timing={};counts={}
    for split in ('dev','test'):
        samples=[s for s in m['samples'] if s['split']==split];rows=f.predictions(DATA/f'{split}.jsonl');check_results(samples,rows)
        metrics[split]={d:{v:f.score([s for s in samples if s['domain']==d],views(rows,v)) for v in ('baseline','assisted')} for d in DOMAINS}
        values=[r['elapsed_ms']['layout'] for r in rows.values()]
        timing[split]={'layout_median_ms':statistics.median(values),'layout_min_ms':min(values),'layout_max_ms':max(values),'ocr_source':'cached original Tiny spans' if split=='dev' else 'actual native Tiny OCR'}
        counts[split]={'images':len(rows),'images_with_regions':sum(bool(r['layout']['regions']) for r in rows.values()),'regions':sum(len(r['layout']['regions']) for r in rows.values()),
            'backfilled_fields':sum(not r['baseline']['fields'][k]['candidates'] and bool(field['candidates']) for r in rows.values() for k,field in r['assisted']['fields'].items())}
        for s in samples:
            row=rows[s['id']]
            for field,label in s['expected'].items():
                actual=row['assisted']['fields'][field]
                if f.key(field,actual['value'],s)!=f.key(field,label,s,label=True):failures.append({'id':s['id'],'split':split,'field':field,'status':'ambiguous' if actual['candidates'] and actual['value'] is None else ('missing' if actual['value'] is None else 'wrong')})
    return {'version':1,'runtime':'RustO 0.3.2 Tiny + optional PP-DocLayout-S; RTen 0.27.0 CPU FP32',
        'protocol':m['protocol'],'profile':profile,'metrics':metrics,'layout_counts':counts,'observed_timing':timing,'failures':failures,
        'prediction_sha256':{name:digest(DATA/name) for name in ('dev.jsonl','test.jsonl')},
        'limits':['Model class regions are not financial field labels. No handwriting model or table structure decoding added.',
            'Community ONNX export with only two NMS constants normalized to official config; no independent Paddle native prediction verification.',
            'Small purposive sample; archival cheque gold manually annotated by implementer, not blinded. Exact original labels retained, no fuzzy credit.',
            'Existing page candidates are never changed; missing-candidate additions require explicit local labels and a review issue. Unchanged aggregate accuracy is a valid result.',
            'Layout timings include decoding/preprocessing/inference. Not a controlled benchmark; development OCR is cached and fresh test OCR is real.']}


def verify_code():
    for cmd in [['cargo','clippy','-p','rs-ocr','--all-features','--all-targets','--locked','--no-deps','--','-D','warnings'],['cargo','fmt','-p','rs-ocr','--','--check']]:
        subprocess.run(['rtk','proxy','mise','exec','--',*cmd],cwd=ROOT,check=True)
    for script in ('financial_benchmark.py','ocr_model_study.py','layout_benchmark.py'):
        subprocess.run(['rtk','proxy','mise','exec','--','python',str(CRATE/'scripts'/script),'verify-report'],cwd=ROOT,check=True)
    verify_history();print('DOC_LAYOUT_CODE_VERIFIED')


def download_model():
    for a in f.read(SPEC)['artifacts']:download(MODEL,a,a['url'])
    subprocess.run(['rtk','proxy','mise','exec','--',str(REFERENCE),str(CRATE/'scripts/doc_layout_reference.py'),'prepare'],cwd=ROOT,check=True)
    verify_artifacts()


def download_data():
    manifest=f.read(MANIFEST)
    old.validate(manifest,[f.read(f.MANIFEST),f.read(CRATE/'model-manifests/ocr-model-samples.json'),f.read(old.MANIFEST)])
    DATA.mkdir(parents=True,exist_ok=True)
    archives={a['path']:a for a in f.read(f.MANIFEST)['sources']['zenodo']['archives']}
    verified=set()
    for sample in manifest['samples']:
        for a in sample['artifacts']:
            if 'url' in a:download(DATA,dict(a,name=a['path']),a['url'])
            else:
                archive=archives[a['archive']]
                if a['archive'] not in verified:
                    download(f.DATA,dict(archive,name=archive['path']),archive['url']);verified.add(a['archive'])
                with zipfile.ZipFile(f.DATA/archive['path']) as z:content=z.read(a['member'])
                if len(content)!=a['size'] or hashlib.sha256(content).hexdigest()!=a['sha256']:raise ValueError('archive member mismatch')
                (DATA/a['path']).write_bytes(content)
    verify_data();print('DOC_LAYOUT_DATA_VERIFIED')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('download-model','download-data','verify-data','parity','verify-model','run-dev','run-test','freeze','report','verify-report','verify-code'));cmd=p.parse_args().command
    if cmd=='download-model':download_model()
    elif cmd=='download-data':download_data()
    elif cmd=='verify-data':verify_data();print('DOC_LAYOUT_DATA_VERIFIED')
    elif cmd in ('parity','verify-model'):model_parity(write=cmd=='parity')
    elif cmd.startswith('run-'):run(cmd[4:])
    elif cmd=='freeze':freeze()
    elif cmd=='report':f.write(REPORT,report());print('DOC_LAYOUT_REPORT_WRITTEN')
    elif cmd=='verify-report':
        if f.read(REPORT)!=report():raise ValueError('stale extraction report')
        print('DOC_LAYOUT_REPORT_VERIFIED')
    else:verify_code()


if __name__=='__main__':main()
