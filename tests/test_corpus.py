from pathlib import Path
from librarian import corpus

def test_highlighted_code_and_cached_utf8_repair(tmp_path, monkeypatch):
    raw=tmp_path/'raw';raw.mkdir()
    html='''<div class="body"><section id="files"><h1>Files</h1>
    <p>It doesn\u2019t need manual closing.</p>
    <pre><span>&gt;&gt;&gt; </span><span>p</span> = <span>Path</span>(<span>'a.txt'</span>)
<span>&gt;&gt;&gt; </span><span>p</span>.<span>read_text</span>()</pre>
    </section></div>'''
    # Reproduce the old HTTP default decoding and the existing on-disk cache.
    (raw/'library_pathlib.html').write_text(html.encode('utf-8').decode('latin1'),encoding='utf-8')
    monkeypatch.setattr(corpus,'DATA',tmp_path)
    monkeypatch.delenv('LIBRARIAN_RAW_DIR',raising=False)
    sections=corpus.download('library/pathlib')
    text=sections[0]['text']
    assert "It doesn\u2019t need manual closing." in text
    assert ">>> p = Path('a.txt')\n>>> p.read_text()" in text
    assert corpus.download('library/pathlib')==sections

def test_correct_unicode_is_not_changed():
    text='It doesn\u2019t change caf\u00e9 text.'
    assert corpus.repair_legacy_encoding(text)==text

def test_adjacent_api_definitions_have_separate_passages(tmp_path, monkeypatch):
    raw=tmp_path/'raw';raw.mkdir()
    html='''<div class="body"><section id="files"><h1>Files</h1>
    <dl class="py method"><dt id="Path.read_text">Path.read_text()</dt>
    <dd><p>Read decoded contents.</p><pre>p.read_text()</pre></dd></dl>
    <dl class="py method"><dt id="Path.write_text">Path.write_text(data)</dt>
    <dd><p>Write text contents.</p><pre>p.write_text('hello')</pre></dd></dl>
    </section></div>'''
    (raw/'library_pathlib.html').write_text(html,encoding='utf-8')
    monkeypatch.setattr(corpus,'DATA',tmp_path)
    monkeypatch.delenv('LIBRARIAN_RAW_DIR',raising=False)
    sections=corpus.download('library/pathlib')
    read=next(s for s in sections if s['symbol']=='Path.read_text')
    write=next(s for s in sections if s['symbol']=='Path.write_text')
    assert read['url'].endswith('#Path.read_text')
    assert 'p.read_text()' in read['text'] and 'write_text' not in read['text']
    assert write['url'].endswith('#Path.write_text')
    assert "p.write_text('hello')" in write['text'] and 'Read decoded' not in write['text']
