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
const verify = new Function('COURSE','esc',`let designKind='ml'; ${functions}; return [renderDesign(),renderTheory(),renderMLCoding(),highlightPython('x = "<script>" # comment')];`);
const [design,theory,coding,highlight] = verify(course,escape);
if (!design.includes('<svg') || !design.includes('framework-list')) throw Error('Missing design diagrams/list');
if (!theory.includes('Bias versus variance') || !coding.includes('LinearRegressor')) throw Error('Missing new material');
if (highlight.includes('<script>') || !highlight.includes('syntax-comment')) throw Error('Highlight escaping failed');
console.log('Passed: design, theory, coding render and safe offline highlighting.');

if (design.includes("data-fw=") || design.includes("Practice Prompts") || design.includes("Key Levers &amp; Tradeoffs")) throw Error("Obsolete design sections remain");
for (const id of ["recommender", "rag-design", "knowledge-agent"]) if (!design.includes(`id="${id}"`)) throw Error(`Missing reference ${id}`);
const codingPage = new Function('COURSE','esc',`let solved=new Set([1]); let lcFilter='all'; ${functions}; return renderCoding();`)(course,escape);
if (!codingPage.includes('CodeSignal Industry Coding Assessment') || codingPage.includes('Implementation practice') || codingPage.includes('Anthropic')) throw Error('Incorrect coding section');
if (course.industryTasks.length !== 7 || course.industryTasks.some(t=>!t.prompt || !t.example)) throw Error('Missing practice prompts');
const shell=await Deno.readTextFile('study-hub/shell.html');
for (const tab of ['practical','companies','stories','leetcode']) if(shell.includes(`data-tab="${tab}"`)) throw Error(`Obsolete tab: ${tab}`);
if (!shell.includes('data-tab="coding"')) throw Error('Coding tab missing');
console.log('Passed: consolidated Coding rendering, seven reconstructed prompts and simplified navigation.');

if (!design.includes('id="model-upgrades"') || design.includes('id="agent-model-evals"')) throw Error("Model upgrades must be nested in knowledge work");

if (!coding.includes("TransformerBlock") || !coding.includes("TransformerBlockNumpy")) throw Error("Missing simple Transformer blocks");

for(const f of [...course.classics,...course.mlFiles,...course.industryTasks]) {
 if(await Deno.readTextFile('docs/'+f.path)!==f.code) throw Error('Published file mismatch: '+f.path);
}
if(html.includes('140bzOcTU637') || html.includes('/Users/wesley')) throw Error('Personal reference in published page');
console.log('Passed: published Python links and privacy checks.');
