"""Verify split backup parts and archived bytes without extracting or running."""
import argparse,hashlib,io,json,tarfile
from pathlib import Path

class Joined(io.RawIOBase):
    def __init__(self,paths):self.paths=iter(paths);self.current=None
    def readable(self):return True
    def readinto(self,buffer):
        while True:
            if self.current is None:
                try:self.current=next(self.paths).open('rb')
                except StopIteration:return 0
            n=self.current.readinto(buffer)
            if n:return n
            self.current.close();self.current=None

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while block:=f.read(1024**2):h.update(block)
    return h.hexdigest()

def main(folder):
    m=json.loads((folder/'manifest.json').read_text());parts=[]
    for info in m['archive_parts']:
        p=folder/info['path'];assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],str(p);parts.append(p)
    expected={x['path']:x for x in m['archived_files']};seen=set()
    with tarfile.open(fileobj=io.BufferedReader(Joined(parts)),mode='r|gz') as archive:
        for member in archive:
            assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            assert member.name not in seen and member.name in expected,member.name
            h=hashlib.sha256();stream=archive.extractfile(member)
            while block:=stream.read(1024**2):h.update(block)
            entry=expected[member.name]
            assert member.size==entry['bytes'] and h.hexdigest()==entry['sha256'],member.name
            seen.add(member.name)
    assert seen==set(expected)
    print(f'PASS: {len(parts)} parts, {len(seen)} archived files; every byte hash matched')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);main(p.parse_args().folder)
