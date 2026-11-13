'use strict';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
}[c]));

const COOKIE_NAME = 'lc75_done';
const STORAGE_FALLBACK = 'lc75_done_fallback';

function loadSolved() {
  const m = document.cookie.match(new RegExp('(?:^|; )' + COOKIE_NAME + '=([^;]*)'));
  let raw = m ? decodeURIComponent(m[1]) : '';
  if (!raw) {
    try { raw = localStorage.getItem(STORAGE_FALLBACK) || ''; } catch {}
  }
  const set = new Set();
  for (const part of raw.split(',')) {
    const n = Number(part);
    if (Number.isInteger(n) && n > 0) set.add(n);
  }
  return set;
}

function saveSolved(set) {
  const csv = Array.from(set).sort((a, b) => a - b).join(',');
  document.cookie = `${COOKIE_NAME}=${encodeURIComponent(csv)}; max-age=31536000; path=/; SameSite=Lax`;
  try { localStorage.setItem(STORAGE_FALLBACK, csv); } catch {}
}

let solved = loadSolved();
let lcFilter = 'all';

function renderFrameworkCards(kind) {
  const fw = COURSE.frameworks[kind];
  const flowHtml = `<div class="flow-strip">${fw.flow.map((step, i) =>
    `${i ? '<span class="flow-arrow">→</span>' : ''}<div class="flow-step">${esc(step.name)}${step.note ? ` <small>· ${esc(step.note)}</small>` : ''}</div>`
  ).join('')}</div>`;

  return `<ol class="framework-list">${fw.steps.map(s => `<li><h3>${esc(s.title)}</h3><ul>${s.items.map(item=>`<li>${item}</li>`).join('')}</ul></li>`).join('')}</ol>`;

}

function renderDesign() {
  return `<div class="section-head"><h2>Design approach</h2><span>Outline upfront in 60s, then choose where to zoom in</span></div>
    ${renderFrameworkCards('ml')}
    ${COURSE.designReferences}`;
}

function renderLeetCode() {
  const allProblems = COURSE.lcGroups.flatMap((g) => g.problems);
  const coreTotal = allProblems.filter((p) => p.tier === 'core').length;
  const coreDone = allProblems.filter((p) => p.tier === 'core' && solved.has(p.n)).length;
  const extTotal = allProblems.filter((p) => p.tier === 'ext').length;
  const extDone = allProblems.filter((p) => p.tier === 'ext' && solved.has(p.n)).length;
  const pct = Math.round(((coreDone + extDone) / allProblems.length) * 100);

  const groupsHtml = COURSE.lcGroups.map((g) => {
    const visible = g.problems.filter((p) => {
      if (lcFilter === 'core') return p.tier === 'core';
      if (lcFilter === 'ext') return p.tier === 'ext';
      if (lcFilter === 'todo') return !solved.has(p.n);
      return true;
    });
    if (!visible.length) return '';
    const gDone = g.problems.filter((p) => solved.has(p.n)).length;
    return `
      <div class="lc-group">
        <div class="lc-group-head">
          <div>${esc(g.category)}</div>
          <span>${gDone} / ${g.problems.length}</span>
        </div>
        ${visible.map((p) => {
          const isDone = solved.has(p.n);
          return `<div class="lc-row ${isDone ? 'done' : ''}">
            <input type="checkbox" data-lc="${p.n}" ${isDone ? 'checked' : ''} aria-label="LC ${p.n} ${esc(p.title)}">
            <span class="lc-num">#${p.n}</span>
            <a class="lc-title" href="https://leetcode.com/problems/${p.slug}/" target="_blank" rel="noopener">${esc(p.title)}</a>
            
            <span class="diff diff-${p.diff}">${p.diff === 'E' ? 'Easy' : p.diff === 'M' ? 'Med' : 'Hard'}</span>
          </div>`;
        }).join('')}
      </div>
    `;
  }).filter(Boolean).join('');

  return `
    <div class="lc-toolbar">
      <div class="lc-stats">
        <span>Core: ${coreDone}/${coreTotal}</span>
        <span>·</span>
        <span>Additional: ${extDone}/${extTotal}</span>
        <div class="lc-bar" aria-hidden="true"><div class="lc-bar-fill" style="width:${pct}%"></div></div>
      </div>
      <div class="lc-filters">
        ${[
          ['all', 'All (75)'],
          ['core', 'Core (55)'],
          ['ext', 'Additional (20)'],
          ['todo', 'Remaining']
        ].map(([k, label]) => `<button class="filter-btn ${lcFilter === k ? 'active' : ''}" data-lcfilter="${k}">${label}</button>`).join('')}
        <button class="reset-btn" id="lc-reset">Reset</button>
      </div>
    </div>
    <div class="grid-2">${groupsHtml}</div>
  `;
}

function highlightPython(source) {
  const re = /#[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:class|def|return|for|in|if|else|elif|while|import|from|as|raise|assert|with|not|and|or|None|True|False|break|continue|try|except|await|async)\b|\b\d+(?:\.\d+)?\b/g;
  let out='', start=0;
  for (const m of source.matchAll(re)) {
    out += esc(source.slice(start,m.index));
    const kind=m[0][0]==='#'?'comment':/["']/.test(m[0][0])?'string':/^\d/.test(m[0])?'number':'keyword';
    out += `<span class="syntax-${kind}">${esc(m[0])}</span>`;
    start=m.index+m[0].length;
  }
  return out+esc(source.slice(start));
}
function renderTheory(){
 return `<h2>ML Theory</h2><p>Answer aloud in 60–90 seconds, then reveal the answer and try the follow-up. These are selected core interview topics, not a measured company-frequency ranking.</p><p>Review formulation, features, evaluation and serving consistency alongside classical ML, deep learning and Transformers.</p>${['ML basics','Deep learning','Transformers'].map(group=>`<h3>${group}</h3>${COURSE.theory.filter(x=>x.group===group).map(x=>`<details class="clean-acc"><summary>${esc(x.q)}</summary><div class="acc-body"><p>${esc(x.a)}</p><p><strong>Follow-up:</strong> ${esc(x.follow)}</p></div></details>`).join('')}`).join('')}<h3>Sources and further reading</h3><ul><li><a href="https://cs229.stanford.edu/summer2022/cs229-notes5.pdf">Stanford CS229: model selection</a> — regularization and validation.</li><li><a href="https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html">Dive into Deep Learning: Transformers</a> — attention and block structure.</li><li><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html">NumPy least squares</a> — rank-deficient systems.</li></ul>`;
}
function renderMLCoding(){
 const notes=[
 '20 minutes. Implement fit/predict with an intercept. Your normal-equation formula is valid with full column rank; n ≥ d+1 alone is insufficient. This version uses lstsq. Follow-up: derive gradient descent or ridge.',
 '25–30 minutes. Keep your assign/update loop. Squared distances avoid sqrt; retain an empty center; stop after a real update. The objective is squared-distance inertia, not mean Euclidean distance. Broadcasting uses O(nkd) temporary memory; your nested loops use less temporary space.',
 '25–30 minutes. Keep your per-class/per-feature counters. Add Laplace smoothing, an unknown category and log scores. fit resets counts. Ties use first-seen class order. Inputs are nonempty, aligned categorical rows; this is not multinomial count-vector NB.',
 '20–25 minutes. Your single-head implementation, with explicit shapes. Follow-up: add a causal mask, then multiple heads. The baseline is intentionally unmasked. The TransformerBlock below reuses it: pre-norm attention + residual, then pre-norm feed-forward + residual. It is a minimal encoder-style block, not a full decoder.',
 '20 minutes. NumPy single-head attention with einsum projections and contractions. Outline the four steps in comments first. Weights are supplied, with no biases; compare against batched matrix multiplication. Use distinct query/key position labels so you compute every token pair. TransformerBlockNumpy adds the same pre-norm/residual structure with a simple ReLU feed-forward network; biases and learned normalization parameters are omitted.'
 ];
 return `<h2>ML Coding</h2><p>Start with these five compact implementations with outline comments and explicit tensor shapes. Attempt one from a blank editor before revealing the reference. Allow 5 minutes for shapes/assumptions, 20–30 for coding, and 10 for tests and a follow-up. No service scaffolding required.</p>${COURSE.classics.map((f,i)=>`<section><h3>${esc(f.label)}</h3><p>${notes[i]}</p><p><code>${esc(f.path)}</code></p><details class="clean-acc"><summary>Reference implementation</summary><div class="acc-body"><pre><code>${highlightPython(f.code)}</code></pre></div></details></section>`).join('')}<details class="clean-acc"><summary>Optional advanced PyTorch and applied AI labs</summary><div class="acc-body">${renderPyTorch()}</div></details>`;
}

function renderPyTorch() {
  return `
    <div class="section-head">
      <h2>PyTorch &amp; Transformer Quick Reference</h2>
    </div>
    <div class="grid-2">
      <div class="fw-card">
        <div class="fw-card-head"><span class="step-num">01</span><h3>Attention Shape Flow</h3></div>
        <ul>
          <li><strong>Split heads:</strong> <code>x.view(B, T, H, D_h).transpose(1, 2)</code> → <code>(B, H, T, D_h)</code></li>
          <li><strong>RoPE + KV Cache:</strong> Rotate <code>q</code> and new <code>k</code> at <code>[past .. past+T_q-1]</code> <em>before</em> concatenating cache on <code>dim=2</code></li>
          <li><strong>GQA expand:</strong> <code>k.repeat_interleave(H_q // H_kv, dim=1)</code></li>
          <li><strong>Merge heads:</strong> <code>out.transpose(1, 2).contiguous().view(B, T, D)</code></li>
        </ul>
      </div>
      <div class="fw-card">
        <div class="fw-card-head"><span class="step-num">02</span><h3>Masking &amp; Serving Math</h3></div>
        <ul>
          <li><strong>Decode causal mask:</strong> <code>k_pos &lt;= past_len + q_pos</code> (not <code>tril(T_q, T_k)</code>)</li>
          <li><strong>All-masked NaN guard:</strong> Replace full-<code>False</code> rows with <code>0.0</code> before <code>softmax</code>, then zero weights</li>
          <li><strong>KV cache / token:</strong> <code>2 × L × H_kv × D_h × bytes</code></li>
          <li><strong>Prefill vs Decode:</strong> Prefill is compute-bound GEMM; decode is HBM bandwidth-bound GEMV</li>
        </ul>
      </div>
      <div class="fw-card theme-purple">
        <div class="fw-card-head"><span class="step-num">03</span><h3>LoRA, QLoRA &amp; DPO</h3></div>
        <ul>
          <li><strong>LoRA:</strong> Freeze <code>W_0</code>; <code>A ~ Kaiming</code>, <code>B = 0</code>; scale by <code>alpha / r</code>; merges with zero serving cost</li>
          <li><strong>QLoRA:</strong> 4-bit <code>NF4</code> base + double quantization + paged optimizers; BF16 compute</li>
          <li><strong>DPO:</strong> <code>-logsigmoid(beta * ((pi_w - pi_l) - (ref_w - ref_l)))</code>; watch verbosity hacking</li>
        </ul>
      </div>
      <div class="fw-card theme-purple">
        <div class="fw-card-head"><span class="step-num">04</span><h3>GRPO &amp; Silent Training Bugs</h3></div>
        <ul>
          <li><strong>GRPO:</strong> No value critic; group advantage <code>(r - mean) / (std + eps)</code>; detach <code>old_logps</code>; guard <code>std = 0</code></li>
          <li><strong>5 classic PyTorch bugs:</strong> Train+val scaler leakage, multiplying pad scores by <code>0</code> instead of <code>-inf</code>, missing <code>zero_grad()</code>, missing <code>model.eval()</code>, raw <code>log(softmax(x))</code></li>
        </ul>
      </div>
    </div>

    <div class="section-head">
      <h2>Local ML &amp; Applied AI Labs</h2>
      <span>Run <code>.venv/bin/pytest ml_coding/ -q</code></span>
    </div>
    ${COURSE.mlFiles.map((f) => `
      <details class="clean-acc">
        <summary><span>${esc(f.label)} · <code>${esc(f.path)}</code></span></summary>
        <div class="acc-body"><pre><code>${highlightPython(f.code)}</code></pre></div>
      </details>
    `).join('')}
  `;
}

function renderCoding() {
  return `<h2>Coding</h2><p>Algorithm practice and progressive implementation exercises. ML algorithms and attention remain in ML Coding.</p>${renderLeetCode()}
  <h2>CodeSignal Industry Coding Assessment</h2><p>Progressive stateful coding practice from collected practice notes. The originals mostly contain implementations and tests rather than complete prompts. The prompts below are reconstructed practice contracts, not official assessment wording. Return values and edge cases are explicit here; the older drafts may differ.</p>
  <p>Attempt one core in 25–35 minutes, then add one extension. Extension bullets are discussion outlines: agree on their full contract before implementation. References are unreviewed drafts with top-level examples, not guaranteed solutions to the reconstructed prompts.</p>
  ${COURSE.industryTasks.map(t=>`<section><h3>${esc(t.title)}</h3><p>${esc(t.prompt)}</p><p><strong>Example:</strong> ${esc(t.example)}</p><details class="clean-acc"><summary>Progressive extensions</summary><div class="acc-body"><ol>${t.extensions.map(x=>`<li>${esc(x)}</li>`).join('')}</ol></div></details><details class="clean-acc"><summary>Collected implementation and tests · spoilers</summary><div class="acc-body"><p><a href="${esc(t.path)}">${esc(t.path)}</a></p><pre><code>${highlightPython(t.code)}</code></pre></div></details></section>`).join('')}
  <details class="clean-acc"><summary>Other coding prompts found in the mixed notes</summary><div class="acc-body"><p>These are separate live-coding themes, not the progressive assessment above. Short practice versions reconstructed from the notes:</p><ul><li><strong>Longest-match tokenizer:</strong> scan left to right, emit the longest vocabulary match, otherwise emit an unknown-token ID and advance one character. For vocabulary app=1, apple=2, appleapp yields [2,1]. Follow-up: trie lookup.</li><li><strong>Web crawler:</strong> given a start URL and a fake getUrls callback, visit each same-host URL once. Test cycles and duplicate links; discuss bounded concurrency afterward.</li><li><strong>File deduplication:</strong> group files with identical contents. Filter by size, then hash in chunks; verify bytes if exact equality is required. Discuss I/O cost and changing files.</li><li><strong>Stack samples to trace:</strong> compare consecutive call stacks, close frames beyond the common prefix, then open new frames. Specify timestamp and final-flush rules before coding.</li></ul></div></details>`;
}

function render() {
  const requested = (location.hash.slice(1) || 'design').toLowerCase();
  const tab = ['leetcode','practical'].includes(requested) ? 'coding' : requested;
  const valid = ['design', 'theory', 'mlcoding', 'coding', 'pytorch'];
  const active = valid.includes(tab) ? tab : 'design';

  document.querySelectorAll('#nav-tabs a').forEach((a) => {
    a.classList.toggle('active', a.dataset.tab === active);
  });

  if (active === 'design') $('#app').innerHTML = renderDesign();
  else if (active === 'coding') $('#app').innerHTML = renderCoding();
  else if (active === 'pytorch' || active === 'mlcoding') $('#app').innerHTML = renderMLCoding();
  else if (active === 'theory') $('#app').innerHTML = renderTheory();
}

document.addEventListener('click', (e) => {
  const jump = e.target.closest('[data-design-jump]');
  if (jump) {
    e.preventDefault();
    document.getElementById(jump.dataset.designJump)?.scrollIntoView({behavior:'smooth'});
    return;
  }
  const fBtn = e.target.closest('[data-lcfilter]');
  if (fBtn) {
    lcFilter = fBtn.dataset.lcfilter;
    $('#app').innerHTML = renderCoding();
    return;
  }
  if (e.target.id === 'lc-reset') {
    if (confirm('Clear all LeetCode checklist progress?')) {
      solved.clear();
      saveSolved(solved);
      $('#app').innerHTML = renderCoding();
    }
  }
});

document.addEventListener('change', (e) => {
  const cb = e.target.closest('[data-lc]');
  if (cb) {
    const num = Number(cb.dataset.lc);
    if (cb.checked) solved.add(num);
    else solved.delete(num);
    saveSolved(solved);
    $('#app').innerHTML = renderCoding();
  }
});

window.addEventListener('hashchange', () => {
  render();
  window.scrollTo(0, 0);
});

render();
