"""Snapshot experiment deliverables without changing live observer files.

Raw captures and recomputable array caches are inventoried, not silently
included in ordinary Git history. Archives preserve original relative paths.
"""
import argparse,gzip,hashlib,io,json,os,re,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OMIT_PARTS={'.git','__pycache__','.pytest_cache','.venv','build','install'}
TEXT={'.json','.yaml','.yml','.py','.md','.txt','.log','.xml','.urdf','.cpp','.hpp','.h','.c','.sh','.cmake','.msg','.srv','.action','.launch'}

class Parts:
    def __init__(self,out,size):self.out=out;self.size=size;self.stream=None;self.count=0;self.used=0;self.files=[]
    def write(self,data):
        total=len(data)
        while data:
            if self.stream is None:
                self.count+=1;self.path=self.out/f'experiment-results.tar.gz.part{self.count:03d}'
                self.stream=self.path.open('xb');self.used=0;self.h=hashlib.sha256()
            take=min(len(data),self.size-self.used);chunk=data[:take];data=data[take:]
            self.stream.write(chunk);self.h.update(chunk);self.used+=take
            if self.used==self.size:self.close_part()
        return total
    def flush(self):
        if self.stream:self.stream.flush()
    def close_part(self):
        if self.stream:
            self.stream.close();self.files.append(dict(path=self.path.name,bytes=self.used,sha256=self.h.hexdigest()));self.stream=None

def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    assert not list(out.glob('experiment-results.tar.gz.part*')),'Do not overwrite an archive snapshot'
    selected=[];excluded=[];skipped=[]
    for folder in ('data','log','runs'):
        for p in sorted((ROOT/folder).rglob('*')):
            if not p.is_file() or any(x in OMIT_PARTS for x in p.relative_to(ROOT).parts):continue
            if p.is_symlink():
                skipped.append(dict(path=str(p.relative_to(ROOT)),reason='symlink; original target files inventoried separately'));continue
            size=p.stat().st_size;rel=p.relative_to(ROOT)
            if a.analysis_streams_only:
                if not (rel.parts[0]=='log' and rel.parts[1].startswith(('joint_lstm_','joint_residual_'))
                        and p.suffix in ('.csv','.npz','.jsonl')):continue
            include=(p.suffix in TEXT or p.suffix in ('.png','.jpg','.jpeg','.pt','.gz')
                     or p.name=='normalization.npz' or (p.suffix=='.csv' and size<=2*1024**2))
            if 'bag' in rel.parts or p.suffix in ('.h5','.db3','.mcap','.bin','.jsonl'):include=False
            if a.analysis_streams_only:include=True
            if include:selected.append(p)
            else:excluded.append(dict(path=str(rel),bytes=size,reason='raw stream or recomputable cache'))
    records=[];writer=Parts(out,48*1024**2)
    with gzip.GzipFile(filename='',mode='wb',fileobj=writer,compresslevel=3,mtime=0) as gz:
        with tarfile.open(mode='w|',fileobj=gz) as tar:
            for i,p in enumerate(selected):
                data=p.read_bytes();rel=p.relative_to(ROOT)
                # Never package credentials if they accidentally appear in a
                # historical command log. Report only a filename on rejection.
                if p.suffix in TEXT and re.search(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}',data):
                    raise RuntimeError('Credential-like material in '+str(rel))
                entry=tarfile.TarInfo(str(rel));entry.size=len(data);entry.mode=0o755 if os.access(p,os.X_OK) else 0o644;entry.mtime=int(p.stat().st_mtime)
                tar.addfile(entry,io.BytesIO(data));records.append(dict(path=str(rel),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
                # Publish all human-readable experiment reports and plots as
                # browsable copies, leaving the originals byte-identical.
                if rel.parts[0]=='log' and p.suffix in ('.md','.png','.jpg','.jpeg') and not any(x.startswith('source_snapshot') or x in ('ros','tmp') for x in rel.parts):
                    target=out/'reports'/Path(*rel.parts[1:]);target.parent.mkdir(parents=True,exist_ok=True)
                    if p.suffix=='.md':
                        def link(m):
                            local=m.group(1)
                            if not local.startswith(str(ROOT)+'/log/'):return m.group(0)
                            source=Path(local);copy=out/'reports'/source.relative_to(ROOT/'log')
                            if source.suffix not in ('.md','.png','.jpg','.jpeg'):return m.group(0)
                            return ']('+os.path.relpath(copy,target.parent)+')'
                        text=re.sub(r'\]\((/home/venom/factr2/[^)]+)\)',link,data.decode())
                        target.write_text(text)
                    else:target.write_bytes(data)
                if i%300==0:print('archived',i,'/',len(selected),flush=True)
    writer.close_part()
    manifest=dict(schema='w3_work_results_snapshot_v1',root=str(ROOT),archive_parts=writer.files,
        archived_files=records,excluded_raw_files=excluded,skipped_symlinks=skipped,
        note='Source/configuration and exported models are committed normally. Concatenate parts and extract at repository root to restore original experiment paths. Live process state is historical evidence, not a resume command.')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(archived_files=len(records),parts=len(writer.files),archive_bytes=sum(x['bytes'] for x in writer.files),excluded_bytes=sum(x['bytes'] for x in excluded)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--analysis-streams-only',action='store_true',help='Supplement: latest joint-LSTM training and live comparison numeric records only')
    main(p.parse_args())
