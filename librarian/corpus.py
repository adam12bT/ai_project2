"""Download official, version-pinned documentation and retain section provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('LIBRARIAN_DATA_DIR', ROOT / 'data'))
VERSION = '3.12'
PAGES = ['tutorial/' + p for p in ['introduction','controlflow','datastructures','modules','inputoutput','errors','classes','stdlib','stdlib2','venv','floatingpoint']]
PAGES += ['library/' + p for p in ['functions','stdtypes','exceptions','string','re','json','csv','pathlib','os','sys','datetime','time','collections','itertools','functools','math','random','statistics','decimal','fractions','typing','dataclasses','enum','argparse','logging','unittest','sqlite3','subprocess','shutil','tempfile','glob','pickle','copy','pprint','textwrap','heapq','bisect','array','queue','threading','concurrent.futures','asyncio','contextlib','abc','io','gzip','zipfile','hashlib','secrets','urllib.request','urllib.parse','http.client','socket','email.message','html','xml.etree.elementtree','configparser','tomllib','venv','traceback','warnings','inspect']]

def repair_legacy_encoding(text):
    """Undo the previous HTTP Latin-1 decoding of UTF-8 documentation caches."""
    try:
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text

def download(page):
    url = f'https://docs.python.org/{VERSION}/{page}.html'
    raw_dir=Path(os.environ.get('LIBRARIAN_RAW_DIR',DATA/'raw'))
    path = raw_dir / (page.replace('/', '_') + '.html')
    if not path.exists():
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        response.encoding = 'utf-8'
        path.write_text(response.text, encoding='utf-8')
    cached = path.read_text(encoding='utf-8')
    html = repair_legacy_encoding(cached)
    if html != cached:
        path.write_text(html, encoding='utf-8')
    soup = BeautifulSoup(html, 'html.parser')
    main = soup.select_one('div.body') or soup.select_one('main')
    if main is None:
        raise ValueError(f'No documentation body: {url}')
    title = main.find('h1').get_text(' ', strip=True).replace('¶', '')
    for unwanted in main.select('.headerlink, .viewcode-link, script, style'):
        unwanted.decompose()
    # Each text block belongs to its closest section. Code indentation is retained.
    groups = {}
    for block in main.find_all(['p', 'pre', 'dt', 'li']):
        if any(parent.name in ('p', 'pre', 'li', 'dt') for parent in block.parents if parent is not main):
            continue
        section = block.find_parent('section')
        anchor = section.get('id', '') if section else ''
        heading = section.find(['h1','h2','h3','h4'], recursive=False) if section else None
        key = (anchor, heading.get_text(' ', strip=True) if heading else title)
        # Highlighted code has a span per token. Adding separators between spans
        # breaks a single Python line into dozens of separate lines.
        text = block.get_text('' if block.name == 'pre' else ' ', strip=block.name != 'pre').strip()
        if text:
            groups.setdefault(key, []).append(text)
    return [dict(page=page, title=title.strip(), section=section, url=url + ('#' + anchor if anchor else ''), version=VERSION, text='\n\n'.join(blocks)) for (anchor, section), blocks in groups.items()]

def build_corpus(words=220, overlap=40):
    if words < 50 or not 0 <= overlap < words:
        raise ValueError('Require words >= 50 and 0 <= overlap < words')
    (DATA / 'raw').mkdir(parents=True, exist_ok=True)
    license_path=DATA/'LICENSE-python-docs.txt'
    if not license_path.exists():
        shared_license=ROOT/'data/LICENSE-python-docs.txt'
        if shared_license.exists():license_path.write_text(shared_license.read_text(encoding='utf-8'),encoding='utf-8')
        else:
            upstream=requests.get('https://docs.python.org/3.12/license.html',timeout=90)
            upstream.raise_for_status()
            license_body=BeautifulSoup(upstream.text,'html.parser').select_one('div.body')
            license_path.write_text('Source: https://docs.python.org/3.12/license.html\n\n'+license_body.get_text('\n',strip=True),encoding='utf-8')
    with ThreadPoolExecutor(max_workers=5) as pool:
        sections = [section for result in pool.map(download, PAGES) for section in result]
    chunks = []
    for section in sections:
        # Split at line boundaries to retain code indentation; oversized lines are windowed.
        lines = section['text'].splitlines(keepends=True)
        buf = []; count = 0; offset = 0
        for line in lines:
            pieces = [line] if len(line.split()) <= words else [' '.join(line.split()[i:i+words])+'\n' for i in range(0,len(line.split()),words)]
            for piece in pieces:
                size = len(piece.split())
                if buf and count + size > words:
                    chunks.append(make_chunk(section, ''.join(buf).strip(), offset)); offset += 1
                    tail = []
                    for previous in reversed(buf):
                        if sum(len(x.split()) for x in tail) + len(previous.split()) > overlap: break
                        tail.insert(0, previous)
                    buf = tail; count = sum(len(x.split()) for x in buf)
                buf.append(piece); count += size
        if buf: chunks.append(make_chunk(section, ''.join(buf).strip(), offset))
    (DATA / 'chunks.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in chunks), encoding='utf-8')
    manifest = dict(version=VERSION, pages=len(PAGES), sections=len(sections), chunks=len(chunks), words=words, overlap=overlap, source='https://docs.python.org/3.12/', license='https://docs.python.org/3.12/license.html', sha256=hashlib.sha256((DATA/'chunks.jsonl').read_bytes()).hexdigest())
    (DATA/'corpus_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))

def make_chunk(section, text, offset):
    return {**{k:v for k,v in section.items() if k != 'text'}, 'id':hashlib.sha256((section['url']+'|'+str(offset)+'|'+text).encode()).hexdigest()[:20], 'text':text}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--words', type=int, default=220); parser.add_argument('--overlap',type=int,default=40)
    args=parser.parse_args(); build_corpus(args.words,args.overlap)
