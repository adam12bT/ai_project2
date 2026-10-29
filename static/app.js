const form = document.querySelector('#form'), q = document.querySelector('#question');
const results = document.querySelector('#results'), statusEl = document.querySelector('#status');
let lastQuestion = '';
function renderAnswer(container, text) {
  const block = document.createElement('div'); block.className = 'generated-answer';
  const appendProse = value => value.split(/\n\s*\n/).filter(p => p.trim()).forEach(value => {
    const paragraph = document.createElement('p'); paragraph.textContent = value.trim(); block.append(paragraph);
  });
  const fences = /```[^\n`]*\n([\s\S]*?)```/g;
  let offset = 0;
  for (const match of text.matchAll(fences)) {
    appendProse(text.slice(offset, match.index));
    const pre = document.createElement('pre'), code = document.createElement('code');
    code.textContent = match[1].replace(/\n$/, ''); pre.append(code); block.append(pre);
    offset = match.index + match[0].length;
  }
  appendProse(text.slice(offset)); container.append(block);
}
fetch('/api/pages').then(r => r.json()).then(pages => pages.forEach(page => {
  const option = document.createElement('option'); option.value = page; option.textContent = page;
  document.querySelector('#page').append(option);
})).catch(() => { statusEl.textContent = 'Could not connect to the server.'; });
document.querySelectorAll('[data-q]').forEach(b => b.onclick = () => { q.value = b.dataset.q; form.requestSubmit(); });
form.onsubmit = async event => {
  event.preventDefault(); document.querySelector('#submit').disabled = true;
  statusEl.textContent = 'Searching the documentation and preparing your answer…'; results.hidden = true;
  try {
    const question = q.value.trim();
    const response = await fetch('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, page: document.querySelector('#page').value || null }) });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed. Please try again.');
    lastQuestion = question;
    const answerEl = document.querySelector('#answer'); answerEl.replaceChildren();
    document.querySelector('#thanks').textContent = '';
    document.querySelector('#timing').textContent = `${Math.round(data.latency_ms)} ms${data.cache_hit ? ' · cached' : ''}`;
    document.querySelector('#results > .note').textContent = ['openai', 'groq'].includes(data.answer_type)
      ? `Generated with ${data.answer_type === 'openai' ? 'OpenAI' : 'Groq'} · ${data.model} · Check the cited sources` : 'Verbatim evidence · Offline mode';
    renderAnswer(answerEl, data.answer);
    for (const c of data.citations) {
      const card = document.createElement('article'); card.className = 'evidence';
      const link = document.createElement('a'); link.href = c.url; link.target = '_blank'; link.rel = 'noopener noreferrer';
      link.textContent = `[${c.number}] ${c.title} →`;
      const section = document.createElement('p'); section.className = 'note'; section.textContent = c.section;
      card.append(link, section);
      for (const text of c.supporting_quotes || []) {
        const quote = document.createElement('blockquote'); quote.textContent = text; card.append(quote);
      }
      const details = document.createElement('details'), summary = document.createElement('summary');
      summary.textContent = 'Read the source passage';
      const passage = document.createElement('pre'); passage.textContent = c.text;
      details.append(summary, passage); card.append(details); answerEl.append(card);
    }
    results.hidden = false; statusEl.textContent = '';
  } catch (error) { statusEl.textContent = error.message; }
  finally { document.querySelector('#submit').disabled = false; }
};
document.querySelectorAll('[data-vote]').forEach(b => b.onclick = async () => {
  try {
    const response = await fetch('/api/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: lastQuestion, helpful: b.dataset.vote === 'yes' }) });
    if (!response.ok) throw new Error(); document.querySelector('#thanks').textContent = 'Thanks — saved.';
  } catch { document.querySelector('#thanks').textContent = 'Could not save. Try again.'; }
});
