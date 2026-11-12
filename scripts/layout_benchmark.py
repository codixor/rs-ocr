"""Frozen Tiny layout study; native Rust inference, immutable historical evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import financial_benchmark as f
from download_clip import digest, download

CRATE, ROOT = f.CRATE, f.ROOT
DATA = CRATE / 'models/layout-public'
MANIFEST = CRATE / 'model-manifests/layout-samples.json'
PROFILE = CRATE / 'model-manifests/layout-profile.json'
REPORT = CRATE / 'docs/layout-results.json'
HISTORY = CRATE / 'model-manifests/layout-history.json'
SOURCES = f.SOURCES + [CRATE / 'src/documents/regions.rs', CRATE / 'examples/region_batch.rs']
DOMAINS = ('receipt', 'invoice', 'cheque')


def sources():
    return {str(p.relative_to(ROOT)): digest(p) for p in SOURCES}


def validate(manifest, prior):
    f.validate_manifest(manifest)
    seen = {s['id'] for m in prior for s in m['samples']}
    hashes = {a['sha256'] for m in prior for s in m['samples'] for a in s['artifacts'] if a['path'] == s['image']}
    for s in manifest['samples']:
        if s['split'] == 'test' and (s['id'] in seen or next(a['sha256'] for a in s['artifacts'] if a['path'] == s['image']) in hashes):
            raise ValueError('fresh test overlaps a prior study')
    for region in manifest.get('regions', []):
        sample = next(s for s in manifest['samples'] if s['id'] == region['id'])
        if sample['domain'] != 'cheque' or len(region['region']) != 4 or any(type(v) is not int or v < 0 for v in region['region']) or not all(region['region'][2:]):
            raise ValueError('invalid caller region')


def verify_data():
    manifest = f.read(MANIFEST)
    validate(manifest, [f.read(f.MANIFEST), f.read(CRATE / 'model-manifests/ocr-model-samples.json')])
    for s in manifest['samples']:
        for a in s['artifacts']:
            p = DATA / a['path']
            if p.stat().st_size != a['size'] or digest(p) != a['sha256']:
                raise ValueError('sample checksum mismatch')
        if s['source'] in ('sroie', 'zenodo'):
            labels = f.read(DATA / (s['id'] + ('.json' if s['source'] == 'sroie' else '.txt')))
            pairs = [('merchant','company'),('date','date'),('total','total')]
            if s['source'] == 'zenodo': pairs += [('tax','iva_amount'),('reference_number','invoice_number')]
            if s['expected'] != {k: labels[v] for k,v in pairs if labels[v]}:
                raise ValueError('published labels changed')
    print('LAYOUT_DATA_VERIFIED')


def download_data():
    archives = {a['path']: a for a in f.read(f.MANIFEST)['sources']['zenodo']['archives']}
    for s in f.read(MANIFEST)['samples']:
        for a in s['artifacts']:
            if 'url' in a:
                download(DATA, dict(a, name=a['path']), a['url'])
            else:
                arc = archives[a['archive']]
                download(f.DATA, dict(arc, name=arc['path']), arc['url'])
                with zipfile.ZipFile(f.DATA / arc['path']) as z: data = z.read(a['member'])
                if len(data) != a['size'] or hashlib.sha256(data).hexdigest() != a['sha256']:
                    raise ValueError('archive content changed')
                (DATA / a['path']).write_bytes(data)
    verify_data()


def check_rows(samples, rows):
    if set(rows) != {s['id'] for s in samples} or any('error' in r for r in rows.values()):
        raise ValueError('incomplete or failed native evidence')
    for r in rows.values():
        for field in r['fields'].values():
            for c in field['candidates']:
                indices = c['span_indices']
                if not indices or any(type(i) is not int or i < 0 or i >= len(r['spans']) for i in indices):
                    raise ValueError('invalid candidate evidence')
                scores = [r['spans'][i]['confidence'] for i in indices]
                score = None if None in scores else min(scores)
                if score != c['confidence']:
                    raise ValueError('candidate confidence differs from weakest evidence')


def rust_batch(example, jobs, output):
    command = ['rtk','proxy','mise','exec','--','cargo','run','-p','rs-ocr','--release','--locked','--example',example,'--',str(CRATE/'models/ppocrv6-tiny')]
    completed = subprocess.run(command, cwd=ROOT, input=''.join(json.dumps(j)+'\n' for j in jobs), text=True, stdout=subprocess.PIPE, check=True)
    # Publish only complete process output, never a partial held-out batch.
    with output.open('x') as dest: dest.write(completed.stdout)


def run(split):
    if PROFILE.exists() and f.read(PROFILE)['sources'] != sources():
        raise ValueError('historical implementation changed; start a new study')
    if split == 'test':
        frozen = f.read(PROFILE)
        if frozen['sources'] != sources() or frozen['manifest_sha256'] != digest(MANIFEST):
            raise ValueError('implementation/data changed after freeze')
    manifest = f.read(MANIFEST)
    samples = [s for s in manifest['samples'] if s['split'] == split]
    baseline = f.predictions(DATA / 'baseline.jsonl')
    output = DATA / f'improved-{split}.jsonl'
    if output.exists():
        if split == 'test': raise ValueError('refusing to overwrite held-out output')
        output.unlink()
    jobs = [dict(baseline[s['id']], decimal_separator=s['decimal_separator']) for s in samples]
    rust_batch('financial_batch', jobs, output)
    check_rows(samples, f.predictions(output))
    print(f'LAYOUT_BATCH_COMPLETED {split}: {len(samples)}')


def freeze():
    if (DATA/'improved-test.jsonl').exists():
        raise ValueError('held-out output exists; cannot refreeze')
    verify_data()
    samples = [s for s in f.read(MANIFEST)['samples'] if s['split'] == 'dev']
    check_rows(samples, f.predictions(DATA/'improved-dev.jsonl'))
    baseline_sources = f.read(DATA/'baseline-sources.json')
    if baseline_sources != f.read(f.PROFILE)['sources']:
        raise ValueError('baseline implementation is not the historical frozen version')
    f.write(PROFILE, {'sources':sources(),'manifest_sha256':digest(MANIFEST),
        'baseline_sources':baseline_sources,'baseline_sha256':digest(DATA/'baseline.jsonl'),
        'development_sha256':digest(DATA/'improved-dev.jsonl'),
        'models': f.read(f.PROFILE)['models'], 'history_sha256':digest(HISTORY)})
    print('LAYOUT_IMPLEMENTATION_FROZEN')


def regions():
    frozen = f.read(PROFILE)
    if frozen['sources'] != sources() or frozen['manifest_sha256'] != digest(MANIFEST):
        raise ValueError('region implementation/data changed after freeze')
    manifest = f.read(MANIFEST)
    jobs=[]
    for region in manifest['regions']:
        sample = next(s for s in manifest['samples'] if s['id'] == region['id'])
        jobs.append(dict(region, image=str(DATA/sample['image'])))
    rust_batch('region_batch', jobs, DATA/'regions.jsonl')
    print('LAYOUT_REGIONS_COMPLETED')


def verify_history():
    for name, sha in f.read(HISTORY).items():
        if digest(CRATE/name) != sha: raise ValueError('historical artifact changed')
    f.verify_sources(f.read(f.PROFILE)['sources'])
    for script in ('financial_benchmark.py','ocr_model_study.py'):
        subprocess.run(['rtk','proxy','mise','exec','--','python',str(CRATE/'scripts'/script),'verify-report'],cwd=ROOT,check=True)
    print('LAYOUT_HISTORY_VERIFIED')


def report():
    verify_data()
    profile = f.read(PROFILE)
    f.verify_sources(profile['sources'], archive=DATA/'frozen-sources')
    if profile['manifest_sha256'] != digest(MANIFEST) or profile['history_sha256'] != digest(HISTORY):
        raise ValueError('frozen layout implementation/data changed')
    if profile['baseline_sha256'] != digest(DATA/'baseline.jsonl') or profile['development_sha256'] != digest(DATA/'improved-dev.jsonl'):
        raise ValueError('baseline/development evidence changed')
    if any(digest(ROOT/p) != sha for p,sha in profile['models'].items()):
        raise ValueError('Tiny model changed')
    manifest = f.read(MANIFEST)
    baseline = f.predictions(DATA/'baseline.jsonl')
    check_rows(manifest['samples'],baseline)
    metrics={}
    failures=[]
    for split in ('dev','test'):
        samples = [s for s in manifest['samples'] if s['split'] == split]
        new = f.predictions(DATA/f'improved-{split}.jsonl')
        check_rows(samples,new)
        if any(baseline[id]['spans'] != row['spans'] for id,row in new.items()):
            raise ValueError('OCR changed during extraction comparison')
        metrics[split]={domain:{'baseline':f.score([s for s in samples if s['domain']==domain],baseline),
            'improved':f.score([s for s in samples if s['domain']==domain],new)} for domain in DOMAINS}
        for s in samples:
            for field,label in s['expected'].items():
                if f.key(field,new[s['id']]['fields'][field]['value'],s) != f.key(field,label,s,label=True):
                    failures.append({'split':split,'id':s['id'],'field':field,
                        'status':'ambiguous' if new[s['id']]['fields'][field]['candidates'] and new[s['id']]['fields'][field]['value'] is None else ('missing' if new[s['id']]['fields'][field]['value'] is None else 'wrong')})
    crop = f.predictions(DATA/'regions.jsonl')
    if set(crop) != {r['id'] for r in manifest['regions']}: raise ValueError('incomplete region diagnostics')
    diagnostics=[]
    for region in manifest['regions']:
        s=next(s for s in manifest['samples'] if s['id']==region['id'])
        row=crop[s['id']]
        diagnostics.append({'id':s['id'],'split':s['split'],'region':region['region'],'error':row.get('error'),
            'full_page_literal_amount':f.target_in_ocr('amount',s['expected']['amount'],s,baseline[s['id']]),
            'region_literal_amount':f.target_in_ocr('amount',s['expected']['amount'],s,row)})
    return {'version':1,'runtime':'Tiny FP32 / RustO 0.3.2 / RTen 0.27.0; unchanged weights',
        'protocol':manifest['protocol'],'profile':profile,'prediction_sha256':{p.name:digest(p) for p in [DATA/'baseline.jsonl',DATA/'improved-dev.jsonl',DATA/'improved-test.jsonl',DATA/'regions.jsonl']},
        'metrics':metrics,'failures':failures,'region_diagnostics':diagnostics,
        'limits':['Small purposive sample. Report positive fields and blank controls separately; no production accuracy claim.',
            'Published labels retained, including known annotation/locale inconsistencies. Exact text uses case/whitespace normalization, no fuzzy credit.',
            'Cheque labels and supplied regions are manual and not blinded. Caller regions are oracle location hints, not automatic field extraction.',
            'Region literal presence ignores layout and is not handwriting accuracy; Tiny can confidently misread handwriting.',
            'Historical reports and original source bytes preserved; older test images are explicitly retired to development.']}


def verify_code():
    for command in [['cargo','clippy','-p','rs-ocr','--all-features','--all-targets','--locked','--','-D','warnings'],['cargo','fmt','-p','rs-ocr','--','--check']]:
        subprocess.run(['rtk','proxy','mise','exec','--',*command],cwd=ROOT,check=True)
    print('LAYOUT_CODE_VERIFIED')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('download-data','verify-data','run-dev','run-test','freeze','regions','report','verify-report','verify-code','verify-history'))
    cmd=p.parse_args().command
    if cmd=='download-data':download_data()
    elif cmd=='verify-data':verify_data()
    elif cmd.startswith('run-'):run(cmd[4:])
    elif cmd=='freeze':freeze()
    elif cmd=='regions':regions()
    elif cmd=='report':f.write(REPORT,report());print('LAYOUT_REPORT_WRITTEN')
    elif cmd=='verify-report':
        if f.read(REPORT)!=report():raise ValueError('stale or changed layout report')
        print('LAYOUT_REPORT_VERIFIED')
    elif cmd=='verify-code':verify_code()
    else:verify_history()


if __name__=='__main__':main()
