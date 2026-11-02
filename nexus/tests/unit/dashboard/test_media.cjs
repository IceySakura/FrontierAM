'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const media = require('../../../nexus/dashboard/static/media.js');
const data = Buffer.from('synthetic bytes').toString('base64');
const uri = `data:image/png;base64,${data}`;
for (const [name, part] of Object.entries({
  chat: {type:'image_url', image_url:{url:uri}},
  responses: {type:'input_image', image_url:uri},
  anthropic: {type:'image', source:{type:'base64', media_type:'image/png', data}},
})) test(`recognizes ${name} without modifying payload`, () => {
  const before = JSON.stringify(part);
  const result = media.inlineImage(part);
  assert.equal(result.src, uri);
  assert.equal(result.mime, 'image/png');
  assert.equal(result.bytes, 15);
  assert.equal(JSON.stringify(part), before);
});
for (const mime of ['png', 'jpeg', 'webp', 'gif']) test(`allows raster ${mime}`, () => {
  assert.ok(media.inlineImage({type:'input_image', image_url:`data:image/${mime};base64,${data}`}).src);
});
for (const source of ['https://example.invalid/image.png', 'javascript:alert(1)', 'file:///secret', '/api/secret', 'data:image/svg+xml;base64,PHN2Zy8+', 'data:text/html;base64,YQ==', 'data:image/png,abc', 'data:image/png;base64,!!!', 'data:image/png;base64,a', 'data:image/png;base64,AA=A', 'data:image/png;base64,AAAA====', 'data:image/png;base64,']) test(`rejects unsafe/invalid source ${source}`, () => {
  const result = media.inlineImage({type:'image_url', image_url:{url:source}});
  assert.ok(result.error); assert.equal(result.src, undefined);
});
test('handles incomplete and unrelated parts without crashing', () => {
  for (const part of [null, 'text', {}, {type:'text', text:'hi'}]) assert.equal(media.inlineImage(part), null);
  for (const part of [{type:'input_image'}, {type:'image_url', image_url:{}}, {type:'image',source:{type:'url',url:'https://example.invalid'}}, {type:'image',source:{type:'base64',media_type:'image/png',data:5}}]) assert.ok(media.inlineImage(part).error);
});
test('handles optional and valid padding', () => {
  for (const encoded of ['YQ==', 'YWI=', 'YWJj', 'YQ', 'YWI']) {
    const info = media.inlineImage({type:'input_image', image_url:`data:image/png;base64,${encoded}`});
    assert.equal(info.bytes, Buffer.from(encoded,'base64').length);
  }
});
test('folds base64 in nested tool results without mutating facts', () => {
  const value = {type:'tool_result', content:[{type:'image', source:{type:'base64',media_type:'image/png',data}}, {type:'image_url',image_url:{url:uri}}], text:'unchanged'};
  const before = JSON.stringify(value);
  const preview = media.compactJSON(value);
  assert.ok(preview.includes('Base64 已折叠'));
  assert.ok(preview.includes('unchanged'));
  assert.ok(!preview.includes(data));
  assert.equal(JSON.stringify(value), before);
  assert.ok(!media.compactJSON(uri).includes(data));
});
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this.classList = {add(){}}; this.isConnected = true; }
  append(...items) { this.children.push(...items); }
  setAttribute(key, value) { this.attributes[key] = value; }
  removeAttribute(key) { delete this[key]; }
}
test('only assigns image src after intersection, then handles decode failure', () => {
  let callback; const watched = new Set();
  global.document = {createElement: tag => new Element(tag)};
  global.IntersectionObserver = class {
    constructor(fn) { callback = fn; }
    observe(image) { watched.add(image); }
    unobserve(image) { watched.delete(image); }
  };
  const figure = media.imageCard(media.inlineImage({type:'input_image',image_url:uri}), 'fixture');
  const [button, caption] = figure.children; const image = button.children[0];
  assert.equal(image.src, undefined);
  callback([{target:image,isIntersecting:false}]); assert.equal(image.src, undefined);
  callback([{target:image,isIntersecting:true}]); assert.equal(image.src, uri);
  image.naturalWidth = 1280; image.naturalHeight = 800; image.onload();
  assert.ok(caption.textContent.includes('1280 × 800'));
  image.onerror(); assert.equal(button.disabled, true); assert.equal(image.src, undefined);
  assert.ok(caption.textContent.includes('解码失败'));
  const other = media.imageCard(media.inlineImage({type:'input_image',image_url:uri}), 'detached');
  const detached = other.children[0].children[0]; detached.isConnected = false;
  media.prune(); assert.equal(watched.size, 0);
});
test('invalid blocks never create an image element', () => {
  const figure = media.imageCard({error:'blocked'}, 'fixture');
  assert.deepEqual(figure.children.map(e => e.tag), ['p']);
});
