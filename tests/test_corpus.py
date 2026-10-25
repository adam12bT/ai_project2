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
