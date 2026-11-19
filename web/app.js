const $ = id => document.getElementById(id);
let mode = 'completion', history = [], busy = false, ready = false, controller = null;
const examples = {
  chat: [['Say hello', '简单打个招呼', 'Hello! What is your name?'], ['Explain the solar system', '试试知识表达', 'What is the solar system?'], ['Talk about education', '试试熟悉的话题', 'Why is education important?'], ['Tell a short story', '试试想象与连贯性', 'Tell me a short story about a cat.']],
  completion: [['Once upon a time…', '续写一个故事', 'Once upon a time, a little cat lived in a small village.'], ['The solar system…', '续写知识类文本', 'The solar system consists of'], ['The purpose of education…', '续写说明性文本', 'The purpose of education is'], ['A simple conversation…', '尝试对话形式的续写', 'User: Hello!\nAssistant: Hello!\nUser: How are you?\nAssistant:']]
};
function error(message = '') { $('error').textContent = message; $('error').hidden = !message; }
function scrollDown() { $('conversation').scrollTop = $('conversation').scrollHeight; }
function setBusy(value) {
  busy = value; $('send').hidden = value; $('stop').hidden = !value;
  $('send').disabled = !ready; $('input').disabled = value;
  ['chatMode', 'completionMode', 'clear', 'reset'].forEach(id => $(id).disabled = value);
  document.querySelectorAll('.suggestion').forEach(el => el.disabled = value);
}
function renderExamples() {
  $('suggestions').replaceChildren();
  for (const [title, description, prompt] of examples[mode]) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'suggestion';
    const strong = document.createElement('strong'); strong.textContent = title;
    const small = document.createElement('small'); small.textContent = description;
    button.append(strong, small); button.addEventListener('click', () => { $('input').value = prompt; $('input').focus(); });
    $('suggestions').append(button);
  }
}
function switchMode(next) {
  if (busy || next === mode) return;
  mode = next; clear();
  $('chatMode').classList.toggle('active', mode === 'chat');
  $('completionMode').classList.toggle('active', mode === 'completion');
  $('pageTitle').textContent = mode === 'chat' ? '对话测试' : '文本续写';
  $('input').placeholder = mode === 'chat' ? '用英文问一个简单的问题…' : '输入一段英文，让模型接着写…';
  $('welcomeDescription').textContent = mode === 'chat' ? '从零开始的 124M 参数。试一个简单的英文问题，看看它学到了什么。' : '用熟悉的英文开头，观察模型的词语选择、知识表达和篇章连贯性。';
  renderExamples();
}
function clear() {
  history = []; $('messages').replaceChildren(); $('welcome').hidden = false; $('input').value = ''; error();
}
function message(role, text) {
  const row = document.createElement('article'); row.className = `message ${role}`;
  const avatar = document.createElement('div'); avatar.className = 'avatar'; avatar.textContent = role === 'user' ? '你' : 'm';
  const body = document.createElement('div'); body.className = 'message-body';
  const label = document.createElement('div'); label.className = 'message-label'; label.textContent = role === 'user' ? '你' : 'GPT-2 small · 你的模型';
  const content = document.createElement('div'); content.className = 'message-text'; content.textContent = text;
  const meta = document.createElement('div'); meta.className = 'message-meta';
  body.append(label, content, meta); row.append(avatar, body); $('messages').append(row); scrollDown();
  return { content, meta };
}
async function submit(event) {
  event.preventDefault();
  const input = $('input').value.trim();
  if (!input || busy || !ready) return;
  const seed = Number($('seed').value);
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) { error('随机种子需要是 0 到 4294967295 之间的整数。'); return; }
  error(); $('welcome').hidden = true; message('user', input);
  const reply = message('assistant', '正在思考…'); reply.content.classList.add('thinking');
  const messages = [...history.slice(-18), { role: 'user', content: input }];
  controller = new AbortController(); setBusy(true); $('input').value = '';
  let answer = '', completed = false, contextNote = '';
  try {
    const response = await fetch('/api/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
      body: JSON.stringify({ mode, messages: mode === 'chat' ? messages : [], prompt: input,
        temperature: Number($('temperature').value), max_new_tokens: Number($('length').value),
        top_k: Number($('topK').value), repetition_penalty: Number($('penalty').value), seed })
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(typeof detail.detail === 'string' ? detail.detail : `请求失败（${response.status}），请检查输入或服务终端。`);
    }
    const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = '';
    function handle(line) {
      if (!line.trim()) return;
      const item = JSON.parse(line);
      if (item.type === 'error') throw new Error(item.message);
      if (item.type === 'start' && item.truncated) contextNote = ' · 上下文已截取最近 512 tokens';
      if (item.type === 'text') { answer = item.text; reply.content.classList.remove('thinking'); reply.content.textContent = answer; scrollDown(); }
      if (item.type === 'done') {
        completed = true; $('speed').textContent = item.tokens_per_second.toFixed(1);
        $('generationMeta').textContent = `${item.tokens} tokens · ${item.seconds.toFixed(2)} 秒`;
        const reason = {length: '达到生成上限，可调长后重试', eos: '模型输出结束符，提前结束', turn_end: '模型切换角色，本轮结束', stopped: '已停止'}[item.finish_reason] || '完成';
        reply.meta.textContent = `${item.tokens} tokens · ${item.tokens_per_second} tok/s · ${reason}${contextNote}`;
      }
    }
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split('\n'); buffer = lines.pop(); lines.forEach(handle);
      if (done) { if (buffer.trim()) handle(buffer); break; }
    }
    if (!completed) throw new Error('连接提前结束，请重试。');
  } catch (err) {
    if (err.name === 'AbortError') reply.meta.textContent = `已停止生成${contextNote}`;
    else { error(err.message || '无法连接模型，请检查本地服务。'); reply.meta.textContent = '本次生成未完成'; }
  } finally {
    reply.content.classList.remove('thinking');
    if (!answer) reply.content.textContent = completed ? '模型直接结束了回复。可以切换到文本续写，或换一个英文开头。' : '没有生成文本。';
    if (answer && mode === 'chat') history = [...messages, { role: 'assistant', content: answer }].slice(-18);
    controller = null; setBusy(false); $('input').focus(); scrollDown();
  }
}
function resetSettings() {
  for (const [id, value] of Object.entries({temperature: 0.8, length: 512, topK: 50, penalty: 1.1, seed: 42})) {
    $(id).value = value; $(id).dispatchEvent(new Event('input'));
  }
}
for (const id of ['temperature', 'length', 'topK', 'penalty']) {
  $(id).addEventListener('input', () => $(id + 'Value').textContent = ['temperature', 'penalty'].includes(id) ? Number($(id).value).toFixed(2) : $(id).value);
}
$('form').addEventListener('submit', submit);
$('input').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('form').requestSubmit(); } });
$('stop').addEventListener('click', () => controller?.abort());
$('clear').addEventListener('click', clear);
$('chatMode').addEventListener('click', () => switchMode('chat'));
$('completionMode').addEventListener('click', () => switchMode('completion'));
$('reset').addEventListener('click', resetSettings);
renderExamples();
fetch('/api/info').then(response => { if (!response.ok) throw new Error(); return response.json(); }).then(info => {
  ready = true; $('connection').textContent = '本地模型已就绪';
  $('deviceBadge').textContent = info.device === 'cuda' ? '● GPU · RTX 推理' : '● CPU 推理';
  $('deviceBadge').title = info.device_name;
  $('trainedTokens').textContent = `${(info.trained_tokens / 1e6).toFixed(1)}M tokens`;
  $('valLoss').textContent = info.validation_loss == null ? '—' : info.validation_loss.toFixed(4);
  $('lossLink').hidden = !info.has_loss_curve; $('send').disabled = false;
  if (info.stage === 'sft') {
    document.querySelector('.pill').textContent = 'SFT MODEL';
    $('modelDescription').textContent = '你的 FineWeb-Edu 基础模型 + 英文对话微调。';
    $('modelNotice').textContent = '这是经过英文对话微调的 124M 模型，建议用英文测试。知识与推理能力仍受基础训练限制。';
    $('lossLabel').textContent = 'SFT 验证 Loss';
    switchMode('chat');
  }
}).catch(() => { $('connection').textContent = '本地服务未连接'; $('connectionDot').classList.add('offline'); $('deviceBadge').textContent = '未连接'; error('无法连接模型。请用 .venv/bin/python web_chat.py 启动服务，再刷新页面。'); });
