'use strict';
const $ = id => document.getElementById(id);
const state = {data: null, mode: 'overview', scale: 1, selected: null, segment: null, collapsed: new Set(), expanded: new Set(), offset: 0, total: 0, query: '', artifact: null, request: 0};
const NS = 'http://www.w3.org/2000/svg';
function el(tag, text, cls) { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; }
function svg(tag, attrs = {}, text) { const n = document.createElementNS(NS, tag); for (const [key, value] of Object.entries(attrs)) n.setAttribute(key, value); if (text !== undefined) n.textContent = text; return n; }
function pretty(value) { return NexusMedia.compactJSON(value); }
function section(parent, title, value) { if (value === undefined || value === null) return; parent.append(el('h3', title), el('pre', pretty(value))); }
async function api(path) { const response = await fetch(path); if (!response.ok) throw new Error(await response.text()); return response.json(); }
function sourceLabel(source) {
  return [source.filename, source.line != null ? `第 ${source.line} 行` : '',
    source.client != null ? `Client ${source.client}` : source.record_path || '',
    source.run_id != null ? `Run ${source.run_id}` : ''].filter(Boolean).join(' · ');
}
function catalogErrors(errors) {
  $('catalog-errors').hidden = !errors.length;
  $('catalog-error-count').textContent = `读取问题 (${errors.length})`;
  const list = $('catalog-error-list'); list.replaceChildren();
  for (const error of errors.slice(0, 100)) list.append(el('p', `${sourceLabel(error)}：${error.error}`));
  if (errors.length > 100) list.append(el('p', '仅展示前 100 条；完整错误列表可从目录 API 获取。'));
}
let catalogRequest = 0;
async function catalog() {
  const request = ++catalogRequest;
  $('catalog-status').textContent = '正在读取目录…';
  try {
    const result = await api(`api/trajectories?offset=${state.offset}&limit=50&q=${encodeURIComponent(state.query)}`);
    if (request !== catalogRequest) return;
    state.total = result.total;
    $('catalog-status').textContent = `${result.total} 条轨迹${result.errors.length ? ` · ${result.errors.length} 条读取问题` : ''}`;
    catalogErrors(result.errors);
    $('catalog').replaceChildren();
    for (const item of result.items) {
      const button = el('button', undefined, 'trajectory-item' + (item.id === state.artifact ? ' selected' : ''));
      button.dataset.id = item.id;
      button.append(el('strong', item.session_id), el('small', `${item.node_count} nodes · ${item.chain_count} chains · ${item.state}`));
      button.append(el('small', sourceLabel(item), 'trajectory-source'));
      button.title = `${item.session_id}\n${sourceLabel(item)}\n${item.created_at || '时间未记录'}`;
      button.onclick = () => openTrajectory(item);
      $('catalog').append(button);
    }
    $('previous').disabled = state.offset === 0;
    $('next').disabled = state.offset + 50 >= result.total;
    if (!state.data && result.items.length) await openTrajectory(result.items[0]);
    if (!result.total) $('catalog').append(el('p', '没有匹配的轨迹。支持原生轨迹 JSON、运行结果 JSON / JSON.gz 和批量 JSONL。', 'muted'));
  } catch (error) { if (request === catalogRequest) $('catalog-status').textContent = `加载失败：${error.message}`; }
}
async function openTrajectory(item) {
  const request = ++state.request;
  $('graph-status').textContent = '正在加载轨迹…';
  try {
    const data = await api(`api/trajectories/${item.id}`);
    if (request !== state.request) return;
    NexusMedia.closePreview();
    state.data = data; state.artifact = item.id; state.selected = null; state.segment = null; state.collapsed.clear(); state.expanded.clear(); state.scale = 1;
    document.querySelectorAll('.trajectory-item').forEach(button => button.classList.toggle('selected', button.dataset.id === item.id));
    $('title').textContent = data.trajectory.session_id;
    $('state').textContent = data.trajectory.state;
    $('subtitle').textContent = `${sourceLabel(item)} · 终态快照 / 只读`;
    $('metrics').replaceChildren();
    const values = [['NODES / 原始节点', data.trajectory.nodes.length], ['CHAINS / 执行流', data.graph.lanes.length], ['FINAL REWARD', data.trajectory.final_reward ?? '未记录'], ['TERMINATION', data.trajectory.termination_reason ?? '未记录']];
    for (const [title, value] of values) { const card = el('div', undefined, 'metric'); card.append(el('span', title), el('strong', String(value))); $('metrics').append(card); }
    $('warnings').textContent = data.graph.warnings.join('\n');
    renderSegments(); showTrajectory(); render();
    $('viewport').scrollTo(0, 0);
  } catch (error) { if (request === state.request) $('graph-status').textContent = `加载失败：${error.message}`; }
}
function showTrajectory() {
  const panel = $('detail'); panel.replaceChildren(el('div', 'TRAJECTORY FACTS', 'eyebrow'), el('h2', '轨迹概要'));
  section(panel, '终态与指标', {state: state.data.trajectory.state, termination_reason: state.data.trajectory.termination_reason, metrics: state.data.trajectory.metrics});
  section(panel, '记录来源', state.data.source);
  section(panel, '运行结果', state.data.run);
  section(panel, '导出说明', state.data.export.omitted);
  panel.append(el('p', '仅展示文件中已记录的事实；不读取外部 token payload，缺失值不会被补造。', 'detail-note'));
  section(panel, 'Extra', state.data.trajectory.extra);
  section(panel, 'Trajectory timeline', state.data.trajectory.timeline); NexusMedia.prune();
}
function showChain(lane) {
  state.selected = `c:${lane.id}`;
  const panel = $('detail'); panel.replaceChildren(el('div', 'CHAIN / EXECUTION STREAM', 'eyebrow'), el('h2', lane.id));
  panel.append(el('p', `${lane.kind} · ${lane.node_ids.length} 个节点`, 'muted'));
  section(panel, 'Chain facts', lane.facts || {note: '默认归属，未声明 Chain，不推断 Agent 类型'});
  panel.append(el('p', 'spawn/join 连到 Chain 入口/出口标记。出口表示整条 Chain 的关联，不代表已知的某个完成消息。', 'detail-note'));
  const toggle = el('button', state.collapsed.has(lane.id) ? '展开泳道' : '折叠泳道', 'details-button');
  toggle.onclick = () => { toggleLane(lane.id); showChain(lane); };
  panel.append(toggle); NexusMedia.prune();
}
function messageCard(node) {
  const card = el('article', undefined, 'message-card');
  const message = node.original_message || {};
  card.append(el('div', `${message.role || node.generated_by} · #${node.node_id}`, 'message-role'));
  let imageCount = 0;
  function content(value, parent, depth = 0) {
    if (depth > 16) { section(parent, '嵌套内容', value); return; }
    if (Array.isArray(value)) { value.forEach(part => content(part, parent, depth + 1)); return; }
    if (value === null || value === undefined) return;
    const image = NexusMedia.inlineImage(value);
    if (image) { parent.append(NexusMedia.imageCard(image, `节点 #${node.node_id} 图片 ${++imageCount}`)); return; }
    if (typeof value === 'object') {
      if (['text', 'input_text', 'output_text'].includes(value.type)) parent.append(el('div', value.text || '', 'message-text'));
      else if (['thinking', 'reasoning'].includes(value.type)) section(parent, 'Thinking', value.thinking || value.text || value);
      else if (value.type === 'tool_result') {
        const result = el('div', undefined, 'tool-result');
        result.append(el('h3', `Tool result · ${value.tool_use_id || ''}${value.is_error ? ' · error' : ''}`));
        content(value.content, result, depth + 1); parent.append(result);
      } else section(parent, value.type || 'Content block', value);
    } else parent.append(el('div', pretty(value), 'message-text'));
  }
  content(message.content, card);
  for (const key of ['reasoning_content', 'thinking', 'tool_calls', 'tool_call_id', 'function_call']) section(card, key, message[key]);
  return card;
}
function rawFacts(node) {
  const raw = el('details'); raw.append(el('summary', `查看节点 #${node.node_id} 原始事实（Base64 折叠）`));
  raw.addEventListener('toggle', () => {
    if (!raw.open || raw.childElementCount > 1) return;
    raw.append(el('pre', pretty(node)));
    const button = el('button', '展开完整 JSON（包含 Base64，可能较大）');
    button.onclick = () => {
      button.disabled = true;
      const pre = el('pre', JSON.stringify(node, null, 2)); raw.append(pre);
      button.textContent = '收起完整 JSON'; button.disabled = false;
      button.onclick = () => { raw.open = false; };
    };
    raw.append(button);
  });
  raw.addEventListener('toggle', () => { if (!raw.open) while (raw.childElementCount > 1) raw.lastChild.remove(); });
  return raw;
}
function selectNode(item) {
  state.selected = item.id;
  const panel = $('detail'); panel.replaceChildren(el('div', 'NODE INSPECTOR', 'eyebrow'), el('h2', `${item.label}${item.node_ids.length > 1 ? ' × ' + item.node_ids.length : ''}`));
  panel.append(el('p', `Chain ${item.lane} · node ${item.node_ids.join(', ')}`, 'muted'));
  if (item.node_ids.length > 1) {
    const expand = el('button', '展开为独立节点', 'details-button'); expand.onclick = () => { state.expanded.add(item.id); render(); }; panel.append(expand);
  }
  for (const id of item.node_ids) {
    const node = state.data.trajectory.nodes.find(n => n.node_id === id);
    panel.append(messageCard(node));
    section(panel, 'Provider / 用量与模型', node.provider_facts);
    section(panel, 'Segment boundary / 已记录证据', node.segment_boundary);
    section(panel, 'Request kwargs', node.request_kwargs);
    section(panel, 'Timeline / 原始时间记录', node.timeline);
    panel.append(rawFacts(node));
  }
  if (state.segment) {
    const back = el('button', `返回 Segment ${state.segment.segment_id}`, 'details-button');
    back.onclick = () => selectSegment(state.segment); panel.append(back);
  }
  panel.scrollTop = 0; NexusMedia.prune();
  document.querySelectorAll('g.node').forEach(node => node.classList.toggle('selected', node.dataset.id === item.id));
}
function segmentRoles(segment) {
  const roles = new Map();
  if (!segment) return roles;
  const targets = new Set(segment.target_node_ids);
  const last = segment.source_node_ids.indexOf(segment.target_node_ids[segment.target_node_ids.length - 1]);
  segment.source_node_ids.forEach((id, index) => roles.set(id, targets.has(id) ? 'target' : index > last ? 'tail' : 'context'));
  return roles;
}
function segmentTokenText(segment) {
  const tokens = segment.tokens;
  return tokens ? `prompt ${tokens.prompt} · response ${tokens.response} · mask-1 ${tokens.generated}` : 'Token 数据未记录或不完整';
}
function segmentNodeRow(label, ids, segment, roles, nodes) {
  const row = el('div', undefined, 'segment-row');
  row.append(el('span', label, 'segment-row-label'));
  const list = el('div', undefined, 'segment-node-list');
  ids.forEach((id, index) => {
    if (index) list.append(el('span', label === 'Targets' ? ' · ' : ' → ', 'segment-separator'));
    const node = nodes.get(id), role = roles.get(id);
    const button = el('button', `n${id}`, `segment-node segment-${role}`);
    button.dataset.nodeId = id;
    button.title = `#${id} · ${node?.original_message.role || node?.generated_by || ''} · ${role === 'target' ? '本段输出' : role === 'tail' ? '尾部观察' : '上下文 / 轮间消息'}`;
    button.setAttribute('aria-label', button.title);
    button.onclick = () => openSegmentNode(segment, id);
    list.append(button);
  });
  if (!ids.length) list.append(el('span', '无', 'muted'));
  row.append(list);
  return row;
}
function renderSegments() {
  const segments = state.data.graph.segments || [], list = $('segments');
  const nodes = new Map(state.data.trajectory.nodes.map(node => [node.node_id, node]));
  list.replaceChildren();
  $('segment-count').textContent = `${segments.length} 段`;
  $('clear-segment').disabled = true;
  if (!segments.length) list.append(el('p', '没有可展示的 Segment（需要已声明 Chain 中的有效 LLM 路径）。', 'muted'));
  segments.forEach((segment, index) => {
    const card = el('article', undefined, 'segment-card'), roles = segmentRoles(segment);
    card.dataset.segmentId = segment.segment_id;
    const heading = el('div', undefined, 'segment-card-heading');
    const choose = el('button', `#${index + 1} · ${segment.segment_id}`, 'segment-select');
    choose.setAttribute('aria-pressed', 'false');
    choose.onclick = () => selectSegment(segment);
    heading.append(choose, el('span', segment.kind, 'segment-kind'));
    card.append(heading, segmentNodeRow('Source', segment.source_node_ids, segment, roles, nodes),
      segmentNodeRow('Targets', segment.target_node_ids, segment, roles, nodes));
    card.append(el('div', segmentTokenText(segment), 'segment-stats'));
    const facts = el('details'); facts.append(el('summary', '分段依据与上下文'));
    const first = segment.first_response_node_index;
    const last = segment.source_node_ids.indexOf(segment.target_node_ids[segment.target_node_ids.length - 1]);
    facts.append(segmentNodeRow('前置上下文', segment.source_node_ids.slice(0, first), segment, roles, nodes),
      segmentNodeRow('尾部观察', segment.source_node_ids.slice(last + 1), segment, roles, nodes));
    if (segment.boundary) section(facts, '开段节点的已记录边界', segment.boundary);
    else facts.append(el('p', '首个输出节点未记录显式边界；这是规范切分得到的起始段。', 'muted'));
    facts.append(el('p', 'Source 可与其他段共享节点。Targets 不是训练启用标记；token 统计不包含最后一次 LLM 后的尾部观察。', 'detail-note'));
    card.append(facts);
    card.onclick = event => { if (!event.target.closest('button,details')) selectSegment(segment); };
    list.append(card);
  });
}
function activateSegment(segment) {
  state.segment = segment; state.selected = null;
  if (state.mode === 'conversation') state.mode = 'overview';
  const ids = new Set(segment.source_node_ids);
  for (const item of state.data.graph.raw.nodes) if (ids.has(item.node_ids[0])) state.collapsed.delete(item.lane);
  document.querySelectorAll('.segment-card').forEach(card => {
    const active = card.dataset.segmentId === segment.segment_id;
    card.classList.toggle('selected', active);
    card.querySelector('.segment-select').setAttribute('aria-pressed', String(active));
  });
  $('clear-segment').disabled = false;
}
function selectSegment(segment) {
  activateSegment(segment); render();
  const panel = $('detail'); panel.replaceChildren(el('div', 'SEGMENT INSPECTOR', 'eyebrow'), el('h2', segment.segment_id));
  panel.append(el('p', `Chain ${segment.chain_id} · ${segment.kind}`, 'muted'));
  section(panel, 'Source / 完整节点列表', segment.source_node_ids);
  section(panel, 'Targets / 本段 LLM 输出', segment.target_node_ids);
  section(panel, '首个输出在 Source 中的下标（从 0 开始）', segment.first_response_node_index);
  section(panel, '开段边界 / 已记录事实', segment.boundary);
  panel.append(el('p', segmentTokenText(segment), 'muted'), el('p', '蓝色为本段输出，灰色为上下文 / 轮间消息，虚线为尾部观察。选中段的聚合节点已展开。', 'detail-note'));
  const clear = el('button', '清除 Segment 高亮', 'details-button'); clear.onclick = clearSegment;
  panel.append(clear); panel.scrollTop = 0; NexusMedia.prune();
}
function openSegmentNode(segment, id) {
  activateSegment(segment); render();
  const item = state.data.graph.raw.nodes.find(node => node.node_ids[0] === id);
  if (item) selectNode(item);
  const group = Array.from(document.querySelectorAll('g.node')).find(node => node.dataset.id === item?.id);
  group?.scrollIntoView({block: 'nearest', inline: 'nearest'});
}
function clearSegment() {
  state.segment = null; state.selected = null;
  document.querySelectorAll('.segment-card').forEach(card => {
    card.classList.remove('selected'); card.querySelector('.segment-select').setAttribute('aria-pressed', 'false');
  });
  $('clear-segment').disabled = true;
  showTrajectory(); render();
}
$('clear-segment').onclick = clearSegment;
function toggleLane(id) { if (state.collapsed.has(id)) state.collapsed.delete(id); else state.collapsed.add(id); render(); }
function displayGraph() {
  const graph = state.data.graph;
  if (state.mode === 'raw') return graph.raw;
  const owners = new Map(), nodes = [], source = new Set(state.segment?.source_node_ids || []);
  for (const group of graph.overview.nodes) {
    if (state.expanded.has(group.id) || group.node_ids.some(id => source.has(id))) {
      for (const id of group.node_ids) { const node = graph.raw.nodes.find(n => n.node_ids[0] === id); nodes.push(node); owners.set(`n:${id}`, node.id); }
    } else { nodes.push(group); group.node_ids.forEach(id => owners.set(`n:${id}`, group.id)); }
  }
  const edges = [], seen = new Set();
  for (const edge of graph.raw.edges) {
    const source = owners.get(edge.source) || edge.source, target = owners.get(edge.target) || edge.target;
    const key = JSON.stringify([source, target, edge.kind]);
    if (source !== target && !seen.has(key)) { edges.push({...edge, source, target}); seen.add(key); }
  }
  return {nodes, edges};
}
function renderConversation() {
  const box = el('div', undefined, 'conversation-list'), select = el('select');
  select.setAttribute('aria-label', '选择 Chain 对话');
  for (const lane of state.data.graph.lanes) { const option = el('option', `${lane.id} · ${lane.kind}`); option.value = lane.id; select.append(option); }
  if (state.segment) select.value = state.segment.chain_id;
  const content = el('div'), roles = segmentRoles(state.segment);
  const update = () => {
    content.replaceChildren(el('p', '按装配顺序展示；存在分支时不是一条线性对话。物理父子关系请切换原始拓扑查看。', 'muted'));
    for (const item of state.data.graph.raw.nodes.filter(n => n.lane === select.value)) {
      const node = state.data.trajectory.nodes.find(n => n.node_id === item.node_ids[0]);
      const card = messageCard(node); card.tabIndex = 0;
      if (state.segment) card.classList.add(`segment-${roles.get(node.node_id) || 'muted'}`);
      card.append(el('small', `parent: ${node.parent ?? '无'} · ${node.selection || '未标记 selection'}`, 'muted'));
      card.onclick = () => selectNode(item); card.onkeydown = e => { if (e.key === 'Enter') selectNode(item); }; content.append(card);
    }
    $('graph-status').textContent = `Chain ${select.value} · 对话${state.segment ? ' · 高亮 ' + state.segment.segment_id : ''}`;
  };
  select.onchange = () => { update(); NexusMedia.prune(); }; box.append(select, content); update(); $('viewport').replaceChildren(box); NexusMedia.prune();
}
function render() {
  if (!state.data) return;
  for (const mode of ['overview', 'raw', 'conversation']) $(mode).classList.toggle('active', state.mode === mode);
  $('zoom-value').textContent = `${Math.round(state.scale * 100)}%`;
  if (state.mode === 'conversation') { renderConversation(); return; }
  const graph = displayGraph(), roles = segmentRoles(state.segment), positions = new Map(), rank = new Map(), indegree = new Map(), outgoing = new Map();
  graph.nodes.forEach(n => {rank.set(n.id, 0); indegree.set(n.id, 0); outgoing.set(n.id, []);});
  for (const edge of graph.edges.filter(e => e.kind === 'physical')) {
    outgoing.get(edge.source)?.push(edge.target); indegree.set(edge.target, (indegree.get(edge.target) || 0) + 1);
  }
  const queue = graph.nodes.filter(n => !indegree.get(n.id)).map(n => n.id);
  for (let i = 0; i < queue.length; i++) for (const target of outgoing.get(queue[i]) || []) {
    rank.set(target, Math.max(rank.get(target), rank.get(queue[i]) + 1)); indegree.set(target, indegree.get(target) - 1); if (!indegree.get(target)) queue.push(target);
  }
  const laneLayouts = [], lanes = state.data.graph.lanes;
  let y = 35, maxX = 720;
  for (const lane of lanes) {
    const items = graph.nodes.filter(n => n.lane === lane.id), occupied = new Map(); let tracks = 1;
    for (const item of items) {
      const column = rank.get(item.id), track = occupied.get(column) || 0; occupied.set(column, track + 1); tracks = Math.max(tracks, track + 1);
      const x = 205 + column * 176; maxX = Math.max(maxX, x + 195);
      if (!state.collapsed.has(lane.id)) positions.set(item.id, {x, y: y + 36 + track * 74, w: 146, h: 52});
    }
    const height = state.collapsed.has(lane.id) ? 84 : Math.max(132, 68 + tracks * 74);
    const columns = items.map(n => rank.get(n.id));
    laneLayouts.push({lane, y, height, start: 181 + (columns.length ? Math.min(...columns) : 0) * 176, end: 369 + (columns.length ? Math.max(...columns) : 0) * 176});
    y += height + 12;
  }
  const width = maxX + 30, height = Math.max(460, y + 35);
  const canvas = svg('svg', {viewBox: `0 0 ${width} ${height}`, width: width * state.scale, height: height * state.scale, role: 'img', 'aria-label': 'Trajectory execution lanes'});
  canvas.dataset.width = width; canvas.dataset.height = height;
  const defs = svg('defs');
  for (const [kind, color] of [['physical', '#bac6d8'], ['spawn', '#6b87d8'], ['join', '#54a28f']]) {
    const marker = svg('marker', {id: `arrow-${kind}`, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse'}); marker.append(svg('path', {d:'M 0 1 L 9 5 L 0 9 z', fill: color})); defs.append(marker);
  }
  canvas.append(defs);
  const endpoints = new Map();
  for (const layout of laneLayouts) {
    const {lane, y, height, start, end} = layout;
    canvas.append(svg('rect', {x: 12, y, width: width - 24, height, rx: 9, fill: lane.id === 'main' ? '#f0f4fd' : '#f5f7fb', opacity: .86}));
    const title = svg('g', {class:'lane-title', role:'button', tabindex:0, 'aria-label': `查看 Chain ${lane.id}`});
    title.append(svg('text', {x:49,y:y+31,fill:'#4f6381','font-size':12,'font-weight':600}, lane.id.length > 19 ? lane.id.slice(0,18) + '…' : lane.id), svg('text', {x:49,y:y+49,fill:'#95a1b4','font-size':9}, `${lane.kind} · ${lane.node_ids.length} nodes`));
    title.onclick = () => showChain(lane); title.onkeydown = e => { if(e.key === 'Enter') showChain(lane); }; canvas.append(title);
    const toggle = svg('text', {x:27,y:y+31,fill:'#7f92ae','font-size':12,role:'button',tabindex:0,class:'lane-title','aria-label':`折叠/展开 ${lane.id}`}, state.collapsed.has(lane.id) ? '▸' : '▾');
    toggle.onclick = () => toggleLane(lane.id); toggle.onkeydown = e => { if(e.key === 'Enter') toggleLane(lane.id); }; canvas.append(toggle);
    endpoints.set(`c:${lane.id}`, {start:{x:start,y:y+62},end:{x:end,y:y+62}});
    for (const [x, label] of [[start,'in'],[end,'out']]) { canvas.append(svg('circle',{cx:x,cy:y+62,r:4,fill:'#b1bfd5'}), svg('text',{x:x-5,y:y+80,fill:'#a2aec0','font-size':8},label)); }
    if (state.collapsed.has(lane.id)) canvas.append(svg('text',{x:start+20,y:y+48,fill:'#8c9ab0','font-size':11},`${lane.node_ids.length} 个节点已折叠 · 关联保留在 Chain 端点`));
  }
  const itemLanes = new Map(graph.nodes.map(item => [item.id, item.lane]));
  for (const edge of graph.edges) {
    const source = positions.get(edge.source), target = positions.get(edge.target);
    const sourceLane = itemLanes.get(edge.source), targetLane = itemLanes.get(edge.target);
    if (edge.kind === 'physical' && sourceLane === targetLane && state.collapsed.has(sourceLane)) continue;
    const sourceEndpoint = endpoints.get(edge.source) || endpoints.get(`c:${sourceLane}`);
    const targetEndpoint = endpoints.get(edge.target) || endpoints.get(`c:${targetLane}`);
    const a = source ? {x:source.x+source.w,y:source.y+source.h/2} : sourceEndpoint?.end;
    const b = target ? {x:target.x,y:target.y+target.h/2} : targetEndpoint?.start;
    if (!a || !b) continue;
    const color = edge.kind === 'spawn' ? '#6b87d8' : edge.kind === 'join' ? '#54a28f' : '#bac6d8';
    const bend = Math.max(30, Math.abs(b.x-a.x)*.42);
    const path = svg('path',{d:`M ${a.x} ${a.y} C ${a.x+bend} ${a.y}, ${b.x-bend} ${b.y}, ${b.x} ${b.y}`,fill:'none',stroke:color,'stroke-width':edge.kind === 'physical' ? 1.2 : 1.7,'marker-end':`url(#arrow-${edge.kind})`});
    if (edge.kind !== 'physical') path.setAttribute('stroke-dasharray','5 4');
    path.append(svg('title',{}, `${edge.kind}: ${edge.source} → ${edge.target}${edge.facts ? '\n'+pretty(edge.facts) : ''}`)); canvas.append(path);
  }
  for (const item of graph.nodes) {
    const pos = positions.get(item.id); if (!pos) continue;
    const boundary = item.boundary, color = boundary ? '#b78131' : item.kind === 'llm' ? '#5b7bd1' : '#7f91aa';
    const role = roles.get(item.node_ids[0]);
    const group = svg('g',{class:`node${state.selected === item.id ? ' selected' : ''}${item.selection === 'rejected_sibling' ? ' rejected' : ''}${state.segment ? ' segment-' + (role || 'muted') : ''}`,role:'button',tabindex:0,'aria-label':`${item.label}, nodes ${item.node_ids.join(',')}${role ? ', segment ' + role : ''}`}); group.dataset.id = item.id;
    group.append(svg('rect',{x:pos.x,y:pos.y,width:pos.w,height:pos.h,rx:7,fill:boundary?'#fffbf2':'#fff',stroke:boundary?'#e6c895':'#dbe3f0'}));
    group.append(svg('circle',{cx:pos.x+14,cy:pos.y+17,r:3,fill:color}));
    const label = boundary ? boundary.kind || 'Segment boundary' : item.label;
    group.append(svg('text',{x:pos.x+25,y:pos.y+21,fill:'#455c7d','font-size':11,'font-weight':550},`${label.slice(0,18)}${item.node_ids.length > 1 ? ' ×'+item.node_ids.length : ''}`));
    group.append(svg('text',{x:pos.x+12,y:pos.y+39,fill:'#95a2b7','font-size':8},`${item.node_ids.length > 1 ? `${item.node_ids.length} messages · 可展开` : '#'+item.node_ids[0]+' · '+(item.selection || item.kind)}`));
    group.append(svg('title',{},`${label}\nnode_ids: ${item.node_ids.join(', ')}`));
    group.onclick = () => selectNode(item); group.onkeydown = e => { if(e.key === 'Enter') selectNode(item); }; canvas.append(group);
  }
  $('viewport').replaceChildren(canvas); NexusMedia.prune();
  $('graph-status').textContent = `${lanes.length} 泳道 · ${graph.nodes.length} 展示节点 / ${state.data.trajectory.nodes.length} 原始节点 · ${graph.edges.filter(e=>e.kind==='spawn').length} spawn · ${graph.edges.filter(e=>e.kind==='join').length} join${state.segment ? ' · 高亮 ' + state.segment.segment_id : ''}`;
}
function zoom(delta) { state.scale = Math.max(.2, Math.min(2.5, state.scale + delta)); render(); }
$('zoom-in').onclick = () => zoom(.15); $('zoom-out').onclick = () => zoom(-.15);
$('fit').onclick = () => { const canvas = $('viewport').querySelector('svg'); if(canvas) { state.scale = Math.max(.2, Math.min(1.2, ($('viewport').clientWidth-10)/Number(canvas.dataset.width))); render(); $('viewport').scrollTo(0,0); } };
for (const mode of ['overview','raw','conversation']) $(mode).onclick = () => { state.mode = mode; render(); };
$('refresh').onclick = catalog; $('previous').onclick = () => { state.offset = Math.max(0,state.offset-50); catalog(); }; $('next').onclick = () => { state.offset += 50; catalog(); };
let debounce;
$('search').oninput = () => { clearTimeout(debounce); debounce = setTimeout(() => { state.query = $('search').value; state.offset = 0; catalog(); }, 200); };
const viewport = $('viewport'); let drag;
viewport.addEventListener('pointerdown', e => { if(e.button !== 0 || e.target.closest('.node,.lane-title,.conversation-list')) return; drag = {x:e.clientX,y:e.clientY,left:viewport.scrollLeft,top:viewport.scrollTop}; viewport.setPointerCapture(e.pointerId); viewport.classList.add('panning'); });
viewport.addEventListener('pointermove', e => { if(drag) { viewport.scrollLeft = drag.left + drag.x-e.clientX; viewport.scrollTop = drag.top + drag.y-e.clientY; } });
for (const event of ['pointerup','pointercancel']) viewport.addEventListener(event, () => { drag=null; viewport.classList.remove('panning'); });
viewport.addEventListener('wheel', e => { if(e.ctrlKey) {e.preventDefault(); zoom(e.deltaY < 0 ? .1 : -.1);} }, {passive:false});
catalog();
