'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function loadApp() {
  const elements = new Map();
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, {append(){}, replaceChildren(){}, addEventListener(){}});
      return elements.get(id);
    },
    createElement() { return {}; },
  };
  const context = vm.createContext({document, fetch: async () => ({ok: true, json: async () => ({items: [], total: 0, errors: []})})});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../../../nexus/dashboard/static/app.js'), 'utf8'), context);
  return context;
}

function evaluate(context, expression) {
  return JSON.parse(vm.runInContext(`JSON.stringify(${expression})`, context));
}

test('catalog source labels distinguish lines and clients without interpreting them as HTML', () => {
  const context = loadApp();
  context.source = {filename: 'batch.jsonl', line: 3, client: '<client>', run_id: 'run-7'};
  assert.equal(vm.runInContext('sourceLabel(source)', context), 'batch.jsonl · 第 3 行 · Client <client> · Run run-7');
  context.source = {filename: 'result.json', record_path: 'processor_results.gateway'};
  assert.equal(vm.runInContext('sourceLabel(source)', context), 'result.json · processor_results.gateway');
  context.source = {filename: 'native.json'};
  assert.equal(vm.runInContext('sourceLabel(source)', context), 'native.json');
});

test('a historical LLM is context in the next segment, not a target', () => {
  const context = loadApp();
  context.segment = {source_node_ids: [0, 1, 2, 3, 4, 5], target_node_ids: [4]};
  assert.deepEqual(evaluate(context, 'Array.from(segmentRoles(segment))'), [[0, 'context'], [1, 'context'], [2, 'context'], [3, 'context'], [4, 'target'], [5, 'tail']]);
  context.segment = {source_node_ids: [0, 1, 2, 3, 4, 5, 6, 7], target_node_ids: [7]};
  assert.equal(vm.runInContext('segmentRoles(segment).get(4)', context), 'context');
  assert.equal(vm.runInContext('segmentRoles(segment).get(5)', context), 'context');
  assert.equal(vm.runInContext('segmentRoles(segment).get(7)', context), 'target');
});

test('multi-turn groups keep all targets and only the final observation is trailing', () => {
  const context = loadApp();
  context.segment = {source_node_ids: [0, 1, 2, 3, 4, 5], target_node_ids: [1, 4]};
  assert.deepEqual(evaluate(context, 'Array.from(segmentRoles(segment))'), [[0, 'context'], [1, 'target'], [2, 'context'], [3, 'context'], [4, 'target'], [5, 'tail']]);
  assert.deepEqual(evaluate(context, 'Array.from(segmentRoles(null))'), []);
});

test('missing token statistics do not masquerade as measured zero', () => {
  const context = loadApp();
  assert.match(vm.runInContext('segmentTokenText({tokens: null})', context), /未记录或不完整/);
  assert.equal(vm.runInContext('segmentTokenText({tokens: {prompt: 0, response: 0, generated: 0}})', context), 'prompt 0 · response 0 · mask-1 0');
});

test('segment highlighting expands shared groups without rewriting physical edges', () => {
  const context = loadApp();
  context.graph = {
    raw: {nodes: [0, 1, 2].map(id => ({id: `n:${id}`, node_ids: [id]})), edges: [
      {source: 'n:0', target: 'n:1', kind: 'physical'}, {source: 'n:1', target: 'n:2', kind: 'physical'},
    ]},
    overview: {nodes: [{id: 'n:0', node_ids: [0, 1]}, {id: 'n:2', node_ids: [2]}]},
  };
  vm.runInContext('state.data = {graph};', context);
  assert.equal(vm.runInContext('displayGraph().nodes.length', context), 2);
  vm.runInContext('state.segment = {source_node_ids: [0, 1, 2], target_node_ids: [1]};', context);
  assert.deepEqual(evaluate(context, 'displayGraph()'), context.graph.raw);
  vm.runInContext('state.mode = "raw";', context);
  assert.deepEqual(evaluate(context, 'displayGraph()'), context.graph.raw);
  vm.runInContext('state.mode = "overview"; state.segment = null;', context);
  assert.equal(vm.runInContext('displayGraph().nodes.length', context), 2);
});
