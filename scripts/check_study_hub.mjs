const katexSource = await Deno.readTextFile('study-hub/vendor/katex/katex.min.js');
const katexModule = {exports:{}};
const katex = new Function('module','exports',katexSource+';return module.exports;')(katexModule,katexModule.exports);
const html = await Deno.readTextFile("docs/index.html");
const match = html.match(/const COURSE = (\{.*?\});\n/s);
if (!match) throw new Error("Missing COURSE payload");
const course = JSON.parse(match[1]);
const count = course.lcGroups.reduce((acc, g) => acc + g.problems.length, 0);
if (count !== 75) throw new Error(`Expected 75 LC problems, got ${count}`);
if (!html.includes("lc75_done")) throw new Error("Missing lc75_done cookie key");
new Function(await Deno.readTextFile("study-hub/app.js"));
console.log(`Passed: minimalist hub verified (${count} LeetCode items, shared design outline and reference designs).`);
const source = await Deno.readTextFile('study-hub/app.js');
const functions = source.slice(source.indexOf('function renderFrameworkCards'), source.indexOf('function render()'));
const escape = s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const verify = new Function('COURSE','esc','katex',`let designKind='ml'; ${functions}; return [renderDesign(),renderTheory(),renderMLCoding(),highlightPython('x = "<script>" # comment')];`);
const [design,theory,coding,highlight] = verify(course,escape,katex);
if (!design.includes('<svg') || !design.includes('framework-list')) throw Error('Missing design diagrams/list');
if (!theory.includes('Bias versus variance') || !coding.includes('LinearRegressor')) throw Error('Missing new material');
if (highlight.includes('<script>') || !highlight.includes('syntax-comment')) throw Error('Highlight escaping failed');
console.log('Passed: design, theory, coding render and safe offline highlighting.');

if (design.includes("data-fw=") || design.includes("Practice Prompts") || design.includes("Key Levers &amp; Tradeoffs")) throw Error("Obsolete design sections remain");
for (const id of ["recommender", "rag-design", "knowledge-agent"]) if (!design.includes(`id="${id}"`)) throw Error(`Missing reference ${id}`);
const codingPage = new Function('COURSE','esc','katex',`let solved=new Set([1]); let lcFilter='all'; ${functions}; return renderCoding();`)(course,escape,katex);
if (/CodeSignal|Industry Coding Assessment|Collected implementation|mixed notes/.test(codingPage)) throw Error('Removed assessment content remains');
if ('industryTasks' in course || html.includes('industry_assessment')) throw Error('Assessment payload or links remain');
for (const path of ['coding/industry_assessment', 'docs/coding/industry_assessment', 'curriculum/industry_assessment.json']) {
  try { await Deno.stat(path); } catch (error) { if (error instanceof Deno.errors.NotFound) continue; throw error; }
  throw Error('Removed assessment files remain: '+path);
}
const shell=await Deno.readTextFile('study-hub/shell.html');
for (const tab of ['practical','companies','stories','leetcode']) if(shell.includes(`data-tab="${tab}"`)) throw Error(`Obsolete tab: ${tab}`);
if (!shell.includes('data-tab="coding"')) throw Error('Coding tab missing');
console.log('Passed: Coding checklist and assessment removal.');

if (!design.includes('id="model-upgrades"') || design.includes('id="agent-model-evals"')) throw Error("Model upgrades must be nested in knowledge work");

if (!coding.includes("TransformerBlock") || !coding.includes("TransformerBlockNumpy")) throw Error("Missing simple Transformer blocks");

for(const f of [...course.classics,...course.mlFiles]) {
 if(await Deno.readTextFile('docs/'+f.path)!==f.code) throw Error('Published file mismatch: '+f.path);
}
if(html.includes('140bzOcTU637') || html.includes('/Users/wesley')) throw Error('Personal reference in published page');
console.log('Passed: published Python links and privacy checks.');

// Theory must work as a readable chapter, including direct section links.
if (!shell.includes('data-tab="design">ML Design</a>')) throw Error('ML Design label missing');
if (/<details\b|<summary\b|<select\b/.test(theory)) throw Error('Theory content is hidden behind controls');
if ((theory.match(/class="theory-topic"/g)||[]).length !== course.theory.length) throw Error('Some theory topics were not rendered');
if (/\\u[0-9a-f]{4}/i.test(JSON.stringify(course.theory))) throw Error('Literal Unicode escape in theory');
const ids = [...theory.matchAll(/ id="([^"]+)"/g)].map(m=>m[1]);
if (new Set(ids).size !== ids.length) throw Error('Duplicate theory section IDs');
for (const [, id] of theory.matchAll(/href="#theory\/([^"]+)"/g)) {
  if (!ids.includes(id)) throw Error('Broken theory contents link: '+id);
}
for (const topic of course.theory) {
  if (!topic.group || !topic.subgroup || !topic.q || !topic.a) throw Error('Incomplete theory topic');
  if (topic.comparison?.rows.some(row=>row.length!==topic.comparison.headers.length)) throw Error('Malformed comparison table');
  for (const ref of topic.sources || []) if (!ref.url.startsWith('https://')) throw Error('Invalid source URL');
}
const renderHelpers = new Function('COURSE','esc','katex',`${functions}; return {renderTheoryTopic,theoryMath,theoryText};`)(course,escape,katex);
const unsafe = renderHelpers.renderTheoryTopic({q:'<script>',a:'<img src=x onerror=alert(1)>'});
if (/<script>|<img/.test(unsafe) || !unsafe.includes('&lt;script&gt;')) throw Error('Theory escaping failed');
if (!renderHelpers.theoryMath(String.raw`\frac{x^2}{\sqrt{d_k}}`).includes('<mfrac>')) throw Error('Typeset fraction missing');
const inlineMath = renderHelpers.theoryText(String.raw`Gradient \(X^\top(p-y)/n\) and <script>`);
if (!inlineMath.includes('class="katex"') || !inlineMath.includes('&lt;script&gt;') || inlineMath.includes('katex-display')) throw Error('Inline math rendering failed');
if (theory.includes('katex-error')) throw Error('Invalid LaTeX in theory');
for (const file of ['katex.min.js','katex.min.css','LICENSE']) {
  if(await Deno.readTextFile('study-hub/vendor/katex/'+file)!==await Deno.readTextFile('docs/vendor/katex/'+file)) throw Error('Published math asset mismatch');
}
const mathCSS = await Deno.readTextFile('docs/vendor/katex/katex.min.css');
for (const [,font] of mathCSS.matchAll(/url\(([^)]+)\)/g)) await Deno.stat('docs/vendor/katex/'+font);
if (!shell.includes('src="vendor/katex/katex.min.js"') || !shell.includes('href="vendor/katex/katex.min.css"')) throw Error('Offline math includes missing');
if (!theory.includes('Kimi Delta Attention') || !theory.includes('InfoNCE') || !theory.includes('WordPiece')) throw Error('Missing expanded theory');

// Exercise the actual hash router with lightweight document stubs, without a browser dependency.
const appNode = {innerHTML:''};
let scrollTarget = null;
const locationStub = {hash:'#theory/theory-tokenizers-transformers-efficient-attention-inference'};
const documentStub = {
  cookie:'', querySelector:()=>appNode, querySelectorAll:()=>[], addEventListener:()=>{},
  getElementById:id=>appNode.innerHTML.includes(`id="${id}"`) ? {scrollIntoView:()=>{scrollTarget=id;}} : null
};
const windowStub = {addEventListener:()=>{},scrollTo:()=>{scrollTarget='top';}};
const runRouter = new Function('COURSE','document','window','location','localStorage','katex',source);
runRouter(course,documentStub,windowStub,locationStub,{getItem:()=>null},katex);
if (scrollTarget !== locationStub.hash.split('/')[1] || !appNode.innerHTML.includes('theory-reader')) throw Error('Direct theory section route failed');
locationStub.hash='#design';
runRouter(course,documentStub,windowStub,locationStub,{getItem:()=>null},katex);
if (!appNode.innerHTML.includes('ML Design approach') || scrollTarget!=='top') throw Error('Design route regressed');
console.log(`Passed: ${course.theory.length} visible theory topics, contents links, direct routes, comparisons and safe math formatting.`);
