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
  return `<ol class="framework-list">${fw.steps.map(s => `<li><h3>${esc(s.title)}</h3><ul>${s.items.map(item=>`<li>${item}</li>`).join('')}</ul></li>`).join('')}</ol>
    <p class="theory-notation"><strong>3-minute pacing check-in:</strong> walk through steps 1–6 high-level in ~2–3 minutes, then pause and ask the interviewer where they want to zoom in—just like an executive summary before a deep dive. Dotted terms (<span class="concept-demo">like this</span>) open quick plain-English explanations inline.</p>`;
}

const CONCEPT_TIPS = {
  'two-tower': {
    title: 'Two-Tower ANN Retrieval',
    body: 'Two separate neural networks—one encodes the user/query into a vector `u`, the other encodes an item into a vector `v` of the same width—trained so relevant pairs have a high dot product `u · v`. Because the item tower never sees the user, you can precompute all 10M item vectors offline overnight and find the closest 500 to `u` in ~10 ms using an Approximate Nearest Neighbor (ANN) index like HNSW or IVF.'
  },
  'id-vs-content': {
    title: 'ID Embeddings vs. Content Embeddings',
    body: 'An **ID embedding** is a giant lookup table where `item_8492` gets its own trainable row of numbers learned purely from who clicked it—great for memorizing viral quirks, completely blind (`all zeros`) on a brand-new item with 0 clicks! **Content embeddings** pass the item’s text, image or metadata through an encoder so new items work on minute one, though they miss fine item-specific popularity signals. Production towers usually combine both.'
  },
  'id-dropout': {
    title: 'ID Dropout & Frequency Thresholds',
    body: 'When you combine `ID_vec + Content_vec`, popular items dominate training logs—so the network lazily memorizes `ID_vec` and leaves the content weights under-trained! **ID dropout** randomly masks `item_id → <UNK>` on 15–30% of training rows (even for viral hits), forcing the model to predict accurately from text/image/metadata alone so cold-start items actually work at serving time.'
  },
  'hash-collisions': {
    title: 'Hash-Trick Collisions & Collisionless Tables',
    body: 'With 500M changing user/item IDs, classic systems map `row = hash(id) % 10_000_000` to cap RAM. That forces ~50 unrelated items to share the exact same embedding row—so a viral video corrupts a niche tutorial’s vector! Collisionless tables (like ByteDance’s *Monolith* cuckoo hashmaps) give active IDs unique slots, ignore 1-time noise IDs (`count < threshold`), and evict stale IDs unseen for 30 days.'
  },
  'inbatch-logq': {
    title: 'In-Batch Negatives & Log-Q Correction',
    body: 'In a minibatch of `B = 1,024` `(user_i, clicked_item_i)` pairs, computing `U @ V.T` gives each user 1,023 "free" negative items from other rows! Catch: a mega-hit item lands in almost every batch and gets pushed down 100× more often than a niche item. Subtracting `log Q(item)` (its log sampling probability) from each logit cancels that popularity penalty mathematically.'
  },
  'embedding-collapse': {
    title: 'Embedding / Dimensional Collapse',
    body: 'A sneaky failure mode where a two-tower or contrastive encoder maps diverse items into a tiny cone or uses only 8 out of 128 vector dimensions (the rest stay correlated or flat), tanking retrieval recall. Spot it early by checking the singular-value spectrum of item vectors and average pairwise cosine similarity of random pairs (which should stay near `0`, not `0.9`).'
  },
  'pit-joins': {
    title: 'Point-in-Time (Time-Travel) Feature Joins',
    body: 'If a user saw an item on Tuesday at 10:00 AM, every feature attached to that training row (`clicks_last_7d`, `account_age`, `item_ctr`) must reflect values known at **Tuesday 10:00 AM**—never Friday’s latest database table! Joining against an unversioned `latest_features` table leaks future clicks into the input and inflates offline AUC dramatically.'
  },
  'chronon-skew': {
    title: 'Write-Once Feature Stores (Chronon / Tecton)',
    body: 'Maintaining SQL/Spark code for training AND separate Go/Java Redis code for serving guarantees subtle null/timezone/window bugs (**online–offline skew**). Declarative platforms let you write a feature once in Python: the engine compiles it into Spark point-in-time historical backfills AND Flink streaming jobs writing to Redis, while auditing live logged values against backfills automatically.'
  },
  'tiled-windows': {
    title: 'Tiled Windows vs. Raw Sliding Windows',
    body: 'Tracking a 30-day exact sliding window in streaming RAM (`Flink + RocksDB`) for 200M users is painfully expensive. Instead, compute a cheap nightly Spark batch count for **days 1–29**, and only maintain **today’s live counter** (or 5-minute non-overlapping tiles) in Flink/Redis—summing `batch_29d + live_today` at read time for 99% of the freshness at ~10% of the streaming cost!'
  },
  'dcnv2': {
    title: 'DCNv2 (Deep & Cross Network v2)',
    body: 'Standard ReLU MLPs are surprisingly bad at simple multiplications like `user_city == restaurant_city` or `price × discount`. A DCNv2 Cross Layer computes `x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l`, explicitly multiplying features back against the original input `x_0` at each step to create clean 2nd- and 3rd-order feature crosses alongside a normal deep MLP.'
  },
  'mmoe-seesaw': {
    title: 'MMoE & the Multi-Task Seesaw',
    body: 'When one shared network predicts partially conflicting goals (clickbait boosts `click` while hurting `dwell` or `hide`), improving one task’s loss often degrades another—the **seesaw effect**. Multi-gate Mixture-of-Experts (**MMoE**) replaces one rigid trunk with 4–8 smaller expert MLPs and gives each task its own learned softmax router to blend experts dynamically (**PLE** also reserves private experts per task).'
  },
  'auction-calibration': {
    title: 'Ad Auctions & Why Calibration Is Monetary',
    body: 'Ranking organic posts only cares about *relative order* (`0.8 > 0.4` gives the exact same order as `0.08 > 0.04`). Ad auctions multiply predicted click rate by real money (`eCPM = Bid × pCTR`) and charge the winner based on the runner-up’s score! If `pCTR` is 2× too high on video ads, video ads unfairly beat organic content and advertisers get overcharged—even though `ROC-AUC` looks perfect.'
  },
  'semantic-ids': {
    title: 'Semantic IDs & Generative Recommenders',
    body: 'Instead of giving an item one arbitrary integer `id = 9841203`, a residual vector quantizer (**RQ-VAE**) compresses its content embedding into a short tuple of shared codebooks, like `[14, 82, 7]` (coarse category → style → fine detail). Items with similar meaning now share prefix tokens, letting a sequence Transformer generate or score recommendations like words in a vocabulary!'
  },
  'ips-shallow-tower': {
    title: 'IPS & Shallow Position Towers (Position Bias)',
    body: 'Users click slot #1 roughly 3–5× more than slot #8 purely because it’s on top. **IPS (Inverse Propensity Weighting)** up-weights rare clicks on slot #8 by `1 / P(examined at #8)` (clipping huge weights to control variance). Even simpler in deep rankers: feed `position` into an isolated **shallow side tower** added to the main logit during training, then drop/fix `position = 1` at serving!'
  },
  'mmr-diversity': {
    title: 'MMR (Maximal Marginal Relevance)',
    body: 'Without diversity rules, a ranker happily fills slots 1–6 with six nearly identical shirts from the same brand. MMR builds the final slate greedily one slot at a time, picking the item maximizing `λ · Relevance(item) − (1 − λ) · max_similarity(item, already_picked)` so each next slot adds fresh variety.'
  },
  'bloom-filter': {
    title: 'Bloom Filter (Seen-Item Suppression)',
    body: 'A compact bit-array using `k` hash functions to answer *"Has this user already seen post X this week?"* in kilobytes instead of megabytes per user. A `0` guarantees they have **never** seen it (zero false negatives); all `1`s means **probably seen** (~1% tunable false-positive rate, which occasionally hides an unseen post—fine for feed deduping, never okay for security ACLs!).'
  },
  'tombstones-acl': {
    title: 'Tombstones & ACL Revocation Lag',
    body: 'Re-chunking and re-embedding a big document can take seconds or minutes in an async queue. If an executive revokes public access on a confidential doc, a **tombstone / authoritative ACL table** immediately blocks that `doc_id` at query time in milliseconds—even before the background vector index finishes deleting the old chunks.'
  },
  'late-chunking': {
    title: 'Late Chunking vs. Pre-Chunking',
    body: 'Standard RAG chops a document into isolated 300-token pieces *before* embedding, so `"It grew 3%"` forgets what `"It"` refers to. **Late chunking** runs the whole 8,000-token document through a long-context bi-encoder *first* (so every token’s hidden state absorbs earlier headings and nouns), and only mean-pools each chunk’s token slice at the final layer!'
  },
  'bm25-vs-dense': {
    title: 'BM25 Lexical Search vs. Dense Vector Search',
    body: '**BM25** is smart keyword search: it matches exact tokens weighted by rarity (IDF) while saturating repeated words and normalizing for document length—unbeatable on exact IDs (`SKU-9921`, `form 10-K`, `NullPointerException`). **Dense search** embeds meaning into continuous vectors—unbeatable on paraphrases (`"car won’t turn over"` ↔ `"dead starter battery"`). Hybrid RAG runs both.'
  },
  'rrf-fusion': {
    title: 'Reciprocal Rank Fusion (RRF)',
    body: 'BM25 outputs unbounded scores like `17.4` while cosine similarity outputs `0.83`, so adding raw scores breaks easily. RRF ignores raw scores completely and sums rank positions: `score(d) = 1 / (60 + rank_bm25(d)) + 1 / (60 + rank_dense(d))`. Anything near the top of *both* lists bubbles cleanly to #1 with zero tuning!'
  },
  'cross-encoder': {
    title: 'Cross-Encoder Reranker',
    body: 'A bi-encoder squashes a whole 300-word chunk into one vector *before* seeing the user’s question, losing fine details like negations or dates. A **cross-encoder** concatenates `[CLS] question [SEP] chunk` and runs full token-to-token self-attention across both—too slow for 10M docs, but blazing fast (~30 ms batched on GPU) to rerank the top 50 retrieved chunks accurately.'
  },
  'prefix-vs-semantic-cache': {
    title: 'Prefix KV Cache vs. Semantic Response Cache',
    body: '**Prefix KV caching** reuses the exact attention Key/Value tensors of static system instructions and tool schemas on the GPU—100% lossless and safe across queries. **Semantic caching** returns a stored answer whenever a new question’s embedding has >0.96 cosine similarity to an old question—dangerous in enterprise apps where *"Q2 margin?"* and *"Q3 margin?"* (or two users with different permissions!) look nearly identical to an embedding.'
  },
  'idempotency-keys': {
    title: 'Idempotency Keys on Write Tools',
    body: 'Suppose an agent calls `send_wire_transfer()` and the network times out after 10 seconds—did the bank execute it before the timeout or not? Passing a deterministic `idempotency_key = "task_88_step_3"` lets the server recognize a retry and return the original receipt instead of sending the money twice!'
  },
  'codeact-tools': {
    title: 'CodeAct / Programmatic Tool Calling',
    body: 'With standard JSON tool calls, pulling 20 tables and filtering rows takes 20 serial LLM turns and dumps megabytes of raw JSON into the prompt. With **CodeAct**, the LLM writes one 12-line Python/JS snippet calling typed tool functions inside a secure sandbox—loops and joins happen in container RAM, returning only the tiny final result to the context window.'
  },
  'cli-vs-mcp': {
    title: 'CLI + Bash vs. MCP / JSON Schemas',
    body: 'Exposing 60 MCP tools stuffs thousands of lines of JSON schema into every prompt and forces the LLM to pass outputs tool-to-tool across turns. Exposing `bash` + CLIs (`gh`, `jq`, `duckdb`, internal CLIs) costs near-zero upfront tokens (the agent runs `mycli --help` only when needed), composes via Unix pipes (`mycli dump > /tmp/a.json && jq ...`), and uses `stderr` exit codes models already know deeply.'
  },
  'brain-hands-split': {
    title: 'Brain / Hands / Session Decoupling',
    body: 'Instead of running the agent loop, credentials, and bash shell inside one container, separate **Brain** (stateless orchestrator calling the LLM), **Session** (append-only durable state log), and **Hands** (replaceable unprivileged sandbox VM). If the VM crashes, Brain spins up a fresh VM; if prompt injection runs `env` inside the VM, zero API keys exist there because authenticated RPCs go through an external proxy!'
  },
  'compaction-handles': {
    title: 'Scoped Handles & Context Compaction',
    body: 'Dumping a 20,000-row SQL result straight into an agent’s message history burns 200k tokens and drowns earlier instructions ("context rot"). Instead, tools store the result behind a typed handle (`table_ref = "tbl_04"`, `rows = 20,412`, `columns = [...]`) so the model queries slices/aggregates as needed, and older turns compact into structured state summaries while keeping raw IDs retrievable.'
  },
  'constrained-decoding': {
    title: 'Constrained Decoding (CFG / FSM Masks)',
    body: 'During token-by-token generation, a finite-state machine tracks where the model currently sits inside your JSON schema (e.g., expecting `"status":` followed by `"open"` or `"closed"`) and sets every illegal token logit to `-∞` before sampling. Guarantees 100% parseable syntax on try #1—never guarantees the chosen value is factually right!'
  },
  'pass-hat-k': {
    title: 'pass^k vs. pass@k (Agent Reliability)',
    body: 'If an agent solves a task on 50% of runs (`p = 0.5`), **`pass@5`** asks *"Did at least 1 of 5 tries succeed?"* (`1 − 0.5⁵ = 97%`—great for offline code contests where a test suite picks the winner!). **`pass^5`** asks *"Did ALL 5 tries succeed?"* (`0.5⁵ = 3%`—what enterprise users actually experience when every flaky failure breaks trust).'
  },
  'ast-chunking': {
    title: 'AST-Aware Code Chunking',
    body: 'Splitting source code every 100 lines chops functions and classes straight down the middle, separating a method’s signature/docstring from its `return` statement. Parsing the file’s Abstract Syntax Tree (via `tree-sitter`) lets an indexer chunk cleanly along whole functions, classes, and imports.'
  },
  'merkle-sync': {
    title: 'Merkle-Tree Codebase Sync',
    body: 'To avoid re-uploading and re-embedding 50,000 files every time a developer switches git branches, the client hashes each file and hashes folder children up to one root hash (a Merkle tree). Comparing client vs. server trees walks *only* down branches whose hashes differ—syncing just the 3 edited files in milliseconds.'
  },
  'speculative-apply': {
    title: 'Speculative Decoding / Fast Apply for Code Edits',
    body: 'Rewriting a 600-line file with only a 5-line fix takes 10+ seconds on a frontier LLM because 99% of tokens are unchanged boilerplate. Using the existing file (or a fast small model’s sketch) as a speculative draft lets the target model verify chunks of 10–30 unchanged tokens in a single parallel GPU forward pass—reaching ~1,000+ tokens/sec with exact output fidelity.'
  },
  'weak-supervision': {
    title: 'Programmatic Weak Supervision & Label Models',
    body: 'Instead of hand-labeling 100k rows, an analyst writes 15 noisy Python labeling functions (`LF_1`: regex match, `LF_2`: domain blocklist, `LF_3`: graph neighbor). Each rule abstains or votes noisily; a **Snorkel label model** estimates each rule’s accuracy and pairwise correlations from agreement rates alone, outputting calibrated probabilistic training labels.'
  },
  'soft-distillation': {
    title: 'Soft-Label Distillation',
    body: 'Training a small fast student on a teacher LLM’s full probability vector (`[0.82 fraud, 0.15 dispute, 0.03 legit]`) teaches rich similarity structure ("this legit edge case looks a lot like a dispute") that a hard `[1, 0, 0]` label throws away—often letting a cheap 10 ms student retain 95%+ of a slow frontier teacher’s accuracy.'
  },
  'active-coreset': {
    title: 'Uncertainty + Core-Set Diversity Active Learning',
    body: 'Asking humans to label *only* the points closest to `p = 0.5` often hands them 200 near-duplicate noisy gibberish rows! **Core-set / cluster diversity** first filters to uncertain candidates, clusters their embeddings, and picks 1 representative per cluster—paired with a 5–10% uniform random sample so novel failure modes where the model is `99%` confidently wrong still get caught.'
  },
  'selective-labels': {
    title: 'Selective-Label (Censorship) Bias',
    body: 'Once a fraud or trust-and-safety model automatically **blocks** transactions with `score > 0.85`, those transactions never complete—so you never observe whether they were real fraud or false declines! Retraining only on approved traffic blinds v2 to blocked patterns. Production systems let a tiny, dollar-capped **0.5–1% randomized control group** through (weighted by inverse propensity) to measure true precision and keep training unbiased.'
  }
};

function renderDesign() {
  return `<div class="design-reader">
    <div class="section-head"><h2>ML Design approach</h2><span>Sketch end-to-end in 3 min, check in, then go deep</span></div>
    ${renderFrameworkCards('ml')}
    ${COURSE.designReferences}
    <div id="concept-popover" class="concept-popover" role="dialog" aria-label="Concept explanation" hidden></div>
  </div>`;
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
function theoryId(text) {
  return 'theory-' + text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/-$/, '');
}

function theoryMath(text, displayMode = true) {
  return katex.renderToString(text, {
    displayMode, throwOnError: true, trust: false,
    strict: 'error', output: 'htmlAndMathml'
  });
}

function formatEscapedProse(safe) {
  return safe
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[\s(])\*([A-Za-z0-9][^*]*?[A-Za-z0-9%])\*($|[\s.,;:!?)—-])/g, '$1<em>$2</em>$3');
}

function theoryText(text) {
  // Explicit inline LaTeX uses KaTeX; ordinary prose is HTML-escaped first, then styled for code/emphasis.
  return String(text ?? '').split(/(\\\(.*?\\\))/g).map(part =>
    part.startsWith('\\(') && part.endsWith('\\)')
      ? theoryMath(part.slice(2, -2), false) : formatEscapedProse(esc(part))
  ).join('');
}

function renderTheoryTopic(topic) {
  const table = topic.comparison;
  return `<article class="theory-topic">
    <h4>${esc(topic.q)}</h4>
    <p>${theoryText(topic.a)}</p>
    <details class="theory-depth"><summary>Explanation, example &amp; follow-ups</summary>
    ${topic.detail ? `<p>${theoryText(topic.detail)}</p>` : ''}
    ${table ? `<div class="clean-table-wrap" role="region" aria-label="${esc(topic.q)} comparison" tabindex="0"><table>
      <thead><tr>${table.headers.map(h=>`<th scope="col">${theoryText(h)}</th>`).join('')}</tr></thead>
      <tbody>${table.rows.map(row=>`<tr>${row.map((cell,i)=>i === 0 ? `<th scope="row">${theoryText(cell)}</th>` : `<td>${theoryText(cell)}</td>`).join('')}</tr>`).join('')}</tbody>
    </table></div>` : ''}
    ${topic.formula ? `<div class="theory-formula" tabindex="0" role="region" aria-label="Formula">${topic.formula.split('\n').map(line=>`<div>${theoryMath(line)}</div>`).join('')}</div>` : ''}
    ${topic.notation ? `<p class="theory-notation">${theoryText(topic.notation)}</p>` : ''}
    ${topic.nuance ? `<p><strong>Nuance:</strong> ${theoryText(topic.nuance)}</p>` : ''}
    ${topic.example ? `<p class="theory-example"><strong>Example:</strong> ${theoryText(topic.example)}</p>` : ''}
    ${topic.follow ? `<p class="theory-follow"><strong>Think it through:</strong> ${theoryText(topic.follow)}</p>` : ''}
    ${topic.sources?.length ? `<p class="theory-sources">Read more: ${topic.sources.map(ref=>`<a href="${esc(ref.url)}">${esc(ref.title)}</a>`).join(' · ')}</p>` : ''}
    </details>
  </article>`;
}

function renderTheory() {
  const groups = [...new Set(COURSE.theory.map(topic=>topic.group))];
  const subgroups = group => [...new Set(COURSE.theory.filter(topic=>topic.group === group).map(topic=>topic.subgroup))];
  return `<div class="theory-reader">
    <h1>ML Theory</h1>
    <p class="theory-intro">Feynman-style progression: model selection → data &amp; drift → evaluation &amp; experiments → optimization → retrieval &amp; ranking → Transformers → scaling &amp; inference → applied LLM evals. Practice saying the plain-English answer aloud in 25 seconds; expand for first-principles intuition, math, mental-math examples and tradeoff tables.</p>
    <nav class="theory-toc" aria-label="Theory contents">
      <h2>Contents</h2>
      <ol>${groups.map(group=>`<li><a href="#theory/${theoryId(group)}">${esc(group)}</a>
        <ul>${subgroups(group).map(sub=>`<li><a href="#theory/${theoryId(group+' '+sub)}">${esc(sub)}</a></li>`).join('')}</ul>
      </li>`).join('')}</ol>
    </nav>
    <p class="theory-notation">Notation: n = examples, d = features, B = batch size, T = sequence length; ᵀ = transpose, ∑ = sum, ‖·‖₂ = Euclidean norm, ⊙ = elementwise product. Symbols are defined locally when their meaning changes.</p>
    ${groups.map((group,index)=>`<section class="theory-section" id="${theoryId(group)}">
      <h2><span class="theory-section-number">${index+1}.</span> ${esc(group)}</h2>
      ${subgroups(group).map(sub=>`<section class="theory-subsection" id="${theoryId(group+' '+sub)}">
        <h3>${esc(sub)}</h3>
        ${COURSE.theory.filter(topic=>topic.group === group && topic.subgroup === sub).map(renderTheoryTopic).join('')}
      </section>`).join('')}
    </section>`).join('')}
    <p class="theory-sources">Foundations reference: <a href="https://cs229.stanford.edu/main_notes.pdf">Stanford CS229 notes</a>. Papers and engineering blogs for modern systems are linked beside the relevant topics.</p>
    <a href="#theory">Back to theory contents ↑</a>
  </div>`;
}
function renderMLCoding(){
 const notes = {
 'linear_regression': '20 minutes. Implement fit/predict with an intercept using a stable least-squares solve (`lstsq`); sample count alone does not guarantee independent columns. Follow-up: gradient descent or ridge (unpenalized intercept).',
 'logistic_regression': '25–30 minutes. Implement overflow-safe sigmoid, signed-logit binary cross entropy, vectorized gradients `X.T @ (p - y) / N` and a fit/predict loop. Follow-up: L2 on `w` (not `b`) or minibatches.',
 'kmeans': '25–30 minutes. Alternate nearest-center assignment (squared distances, no `sqrt`) and centroid mean updates. Retain empty centers and compute squared-distance inertia. Follow-up: `||x||^2 + ||c||^2 - 2 x @ c.T` memory trick or k-means++.',
 'naive_bayes': '25–30 minutes. Categorical counts, Laplace smoothing with a reserved unknown category and log-space scoring (`fit` resets counts). Follow-up: normalized probabilities via log-sum-exp.',
 'pooling': '20 minutes. Masked mean and max pooling over variable-length token sequences `(B, T, D)`. Fill padded tokens with `-inf` before max so negative embeddings beat padding, and guard all-padding rows.',
 'cosine_retrieval': '20–25 minutes. L2-normalize query and catalog vectors safely (zero vectors score 0), compute inner products and return deterministic top-k indices/scores. Follow-up: `argpartition` for `O(N + k log k)`.',
 'ranking_metrics': '15–20 minutes. Implement `roc_auc` via a direct descending ROC curve trapezoid walk (handles score ties cleanly with zero memorized rank algebra) plus `ndcg_at_k` with an `IDCG == 0` guard.',
 'self_attention_numpy': '20–25 minutes for attention only. Project Q/K/V with explicit `einsum` labels, compute scaled scores, normalize over keys and combine values. `TransformerBlockNumpy` is an optional follow-up.',
 'tabular_gbdt_baseline': '30–40 min Jupyter / practical classification round. Get an end-to-end pipeline running fast with zero imputers or one-hot encoders: split first, cast string columns to `category`, fit `LGBMClassifier`, print train vs. test accuracy + precision/recall/F1 + ROC-AUC, then iterate.'
 };
 const appliedAi = COURSE.mlFiles.filter(f => f.path.includes('applied_ai_live') || f.path.includes('ex6_ai_live'));
 return `<h2>ML Coding</h2><p>Focus on readable, working 20–35 minute implementations—nothing fancy or hard to memorize. Clarify shapes and edge cases aloud, write the simplest clean baseline first, verify on a tiny example, then layer on follow-ups.</p>
 <p class="theory-notation">Suggested order: linear regression → logistic regression → K-means → Naive Bayes → pooling → cosine retrieval → ROC-AUC &amp; NDCG@K → attention &amp; MHA → practical GBDT tabular baseline.</p>
 ${COURSE.classics.map(f=>`<section><h3>${esc(f.label)}</h3><p>${theoryText(notes[f.path.split('/').pop().replace('.py','')])}</p><p><a href="${esc(f.path)}" download>Download Python</a> · <code>${esc(f.path)}</code></p><details class="clean-acc"><summary>Reference implementation</summary><div class="acc-body"><pre><code>${highlightPython(f.code)}</code></pre></div></details></section>`).join('')}
 <section><h3>Practice with and without AI</h3><ul>
 <li><strong>1. Lock the contract first:</strong> state tensor/data shapes, complexity target and edge cases (ties, empty inputs, padding, zero norms, train/val split) aloud <em>before</em> writing or generating code.</li>
 <li><strong>2. Delegate bounded pieces (AI-enabled rounds):</strong> use the assistant for boilerplate or unit-test scaffolding while keeping ownership of core algorithm choices.</li>
 <li><strong>3. Audit line-by-line:</strong> check generated code for silent ML traps—masking with <code>0</code> instead of <code>-inf</code>, train-to-val leakage, mutating caller data, or swallowing non-timeout exceptions.</li>
 <li><strong>4. Drive production follow-ups:</strong> discuss streaming/iterator inputs, batching/memory bounds, concurrency/idempotency and monitoring.</li>
 </ul></section>
 <details class="clean-acc"><summary>Applied AI live-coding mocks (plain Python: chunking, async RAG audit, ACL retrieval, bounded agent loop, eval harness)</summary><div class="acc-body"><p>Dependency-free 25–40 minute Python exercises mirroring real Applied AI screens (code review + bug repair, permission pre-filtering, deterministic tool-calling loops and streaming retrieval evaluation). Run locally with <code>python3 -m pytest ml_coding/foundational_labs/ -q</code>.</p>${appliedAi.map(f=>`<details class="clean-acc"><summary><span>${esc(f.label)} · <code>${esc(f.path)}</code></span></summary><div class="acc-body"><p><a href="${esc(f.path)}" download>Download Python</a></p><pre><code>${highlightPython(f.code)}</code></pre></div></details>`).join('')}</div></details>
 <details class="clean-acc"><summary>Optional PyTorch reference and Transformer / post-training labs</summary><div class="acc-body">${renderPyTorch()}</div></details>`;
}

function renderPyTorch() {
  const pytorchFiles = COURSE.mlFiles.filter(f => !f.path.includes('applied_ai_live') && !f.path.includes('ex6_ai_live'));
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
      <h2>Local PyTorch Labs</h2>
      <span>Run <code>.venv/bin/pytest ml_coding/pytorch_transformers/ -q</code></span>
    </div>
    ${pytorchFiles.map((f) => `
      <details class="clean-acc">
        <summary><span>${esc(f.label)} · <code>${esc(f.path)}</code></span></summary>
        <div class="acc-body"><pre><code>${highlightPython(f.code)}</code></pre></div>
      </details>
    `).join('')}
  `;
}

function renderCoding() {
  return `<h2>Coding</h2><p>Algorithm practice with the LeetCode checklist. ML algorithms and attention implementations are in ML Coding.</p>${renderLeetCode()}`;
}

function render() {
  const requested = (location.hash.slice(1).split('/')[0] || 'design').toLowerCase();
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
  const sectionId = ['theory', 'design'].includes(active) ? location.hash.split('/')[1] : null;
  const section = sectionId ? document.getElementById(sectionId) : null;
  if (section) section.scrollIntoView();
  else window.scrollTo(0, 0);
}

let activeTipBtn = null;
let pinnedTip = false;
let hideTipTimer = null;

function closeConceptTip() {
  clearTimeout(hideTipTimer);
  const pop = document.getElementById('concept-popover');
  if (pop) pop.hidden = true;
  if (activeTipBtn) activeTipBtn.classList?.remove('is-open');
  activeTipBtn = null;
  pinnedTip = false;
}

function showConceptTip(btn, pin = false) {
  const entry = CONCEPT_TIPS[btn.dataset.tip];
  const pop = document.getElementById('concept-popover');
  if (!entry || !pop) return;
  clearTimeout(hideTipTimer);
  if (activeTipBtn && activeTipBtn !== btn) activeTipBtn.classList.remove('is-open');
  activeTipBtn = btn;
  if (pin) pinnedTip = true;
  btn.classList.toggle('is-open', pinnedTip);

  pop.innerHTML = `<div class="concept-popover-head"><strong>${esc(entry.title)}</strong><button type="button" class="concept-popover-close" aria-label="Close">×</button></div><p>${theoryText(entry.body)}</p>`;
  pop.hidden = false;

  const rect = btn.getBoundingClientRect();
  const popRect = pop.getBoundingClientRect();
  const vw = window.innerWidth || 1024;
  const vh = window.innerHeight || 768;
  let left = rect.left + rect.width / 2 - popRect.width / 2;
  left = Math.max(12, Math.min(left, vw - popRect.width - 12));
  let top = rect.top - popRect.height - 10;
  if (top < 64) top = Math.min(vh - popRect.height - 12, rect.bottom + 10);
  pop.style.left = `${Math.round(left)}px`;
  pop.style.top = `${Math.round(Math.max(12, top))}px`;
}

document.addEventListener('mouseover', (e) => {
  const term = e.target.closest?.('[data-tip]');
  if (term) {
    if (!pinnedTip || activeTipBtn === term) showConceptTip(term, false);
    return;
  }
  if (e.target.closest?.('#concept-popover')) clearTimeout(hideTipTimer);
});

document.addEventListener('mouseout', (e) => {
  if (pinnedTip) return;
  if (e.target.closest?.('[data-tip]') || e.target.closest?.('#concept-popover')) {
    clearTimeout(hideTipTimer);
    hideTipTimer = setTimeout(closeConceptTip, 150);
  }
});

document.addEventListener('focusin', (e) => {
  const term = e.target.closest?.('[data-tip]');
  if (term && !pinnedTip) showConceptTip(term, false);
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeConceptTip();
});

window.addEventListener('scroll', () => {
  if (activeTipBtn && !pinnedTip) closeConceptTip();
}, { passive: true });

document.addEventListener('click', (e) => {
  if (e.target.closest?.('.concept-popover-close')) {
    closeConceptTip();
    return;
  }
  const term = e.target.closest?.('[data-tip]');
  if (term) {
    e.preventDefault();
    if (pinnedTip && activeTipBtn === term) closeConceptTip();
    else showConceptTip(term, true);
    return;
  }
  if (activeTipBtn && !e.target.closest?.('#concept-popover')) {
    closeConceptTip();
  }
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
});

render();
