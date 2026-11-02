'use strict';
(() => {
  const types = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/gif']);
  function inlineImage(part) {
    if (!part || typeof part !== 'object') return null;
    let uri, mime, data;
    if (part.type === 'image_url' || part.type === 'input_image') {
      uri = typeof part.image_url === 'string' ? part.image_url : part.image_url?.url;
    } else if (part.type === 'image' && part.source?.type === 'base64') {
      mime = part.source.media_type;
      data = part.source.data;
    } else if (part.type === 'image') {
      return {error: '此图片不是内嵌 Base64；本版不加载外部 URL 或文件。'};
    } else return null;
    if (uri !== undefined) {
      if (typeof uri !== 'string' || !uri.startsWith('data:')) return {error: '外部图片未加载；本版仅展示内嵌 Base64。'};
      const comma = uri.indexOf(',');
      const header = uri.slice(0, comma).match(/^data:(image\/[a-z0-9.+-]+);base64$/i);
      if (!header || comma < 0) return {error: '图片 Data URI 格式无效。'};
      mime = header[1]; data = uri.slice(comma + 1);
    }
    if (typeof mime !== 'string' || !types.has(mime.toLowerCase())) return {error: '不支持的图片类型；仅支持 PNG、JPEG、WebP、GIF，不渲染 SVG。'};
    if (typeof data !== 'string' || !data.length || data.length % 4 === 1 || /[^A-Za-z0-9+/=]/.test(data)) return {error: '图片 Base64 数据无效。'};
    const padding = data.endsWith('==') ? 2 : data.endsWith('=') ? 1 : 0;
    if (data.slice(0, data.length - padding).includes('=') || (padding && data.length % 4 !== 0)) return {error: '图片 Base64 填充无效。'};
    return {src: uri || `data:${mime};base64,${data}`, mime: mime.toLowerCase(), bytes: Math.floor(data.length * 3 / 4) - padding};
  }
  function compactJSON(value) {
    const replacer = function(key, item) {
      if (typeof item !== 'string') return item;
      if (item.startsWith('data:') && item.slice(0, 120).includes(';base64,')) {
        return item.slice(0, item.indexOf(',') + 1) + `[Base64 已折叠 · ${item.length} 字符]`;
      }
      if (key === 'data' && this?.type === 'base64') return `[Base64 已折叠 · ${item.length} 字符]`;
      return item;
    };
    return typeof value === 'string' ? replacer.call({}, '', value) : JSON.stringify(value, replacer, 2);
  }
  function element(tag, text, cls) {
    const result = document.createElement(tag);
    if (text !== undefined) result.textContent = text;
    if (cls) result.className = cls;
    return result;
  }
  let observer;
  const pending = new Map();
  function watch(image, src) {
    if (!('IntersectionObserver' in globalThis)) { image.src = src; return; }
    if (!observer) observer = new IntersectionObserver(entries => {
      for (const entry of entries) if (entry.isIntersecting) {
        const source = pending.get(entry.target);
        if (source) entry.target.src = source;
        pending.delete(entry.target); observer.unobserve(entry.target);
      }
    }, {rootMargin: '160px'});
    pending.set(image, src); observer.observe(image);
  }
  function prune() {
    for (const image of pending.keys()) if (!image.isConnected) { observer.unobserve(image); pending.delete(image); }
  }
  function closePreview() { document.querySelector('.image-dialog')?.close(); }
  function preview(info, label, trigger) {
    closePreview();
    const dialog = element('dialog', undefined, 'image-dialog');
    dialog.setAttribute('aria-label', label);
    const bar = element('div', undefined, 'image-dialog-bar');
    const close = element('button', '关闭 · Esc'); close.type = 'button';
    close.onclick = () => dialog.close();
    bar.append(element('strong', label), close);
    const image = element('img'); image.alt = label; image.src = info.src;
    image.onerror = () => { image.replaceWith(element('p', '图片解码失败，原始数据仍保留在轨迹中。', 'image-error')); };
    dialog.append(bar, image);
    dialog.addEventListener('close', () => { image.removeAttribute('src'); dialog.remove(); if (trigger.isConnected) trigger.focus(); });
    dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
    document.body.append(dialog); dialog.showModal(); close.focus();
  }
  function imageCard(info, label) {
    const figure = element('figure', undefined, 'inline-image');
    if (info.error) { figure.append(element('p', info.error, 'image-error')); return figure; }
    const button = element('button', undefined, 'image-thumbnail'); button.type = 'button';
    button.setAttribute('aria-label', `放大${label}`);
    const image = element('img'); image.alt = label; image.decoding = 'async'; image.loading = 'lazy';
    const caption = element('figcaption', `${info.mime} · ${(info.bytes / 1024).toFixed(1)} KiB · 点击放大`);
    image.onload = () => { caption.textContent = `${image.naturalWidth} × ${image.naturalHeight} · ${info.mime} · ${(info.bytes / 1024).toFixed(1)} KiB · 点击放大`; };
    image.onerror = () => { button.disabled = true; image.hidden = true; image.removeAttribute('src'); caption.textContent = '图片解码失败，原始数据仍保留在轨迹中。'; caption.classList.add('image-error'); };
    button.onclick = event => { event.stopPropagation(); preview(info, label, button); };
    button.onkeydown = event => event.stopPropagation();
    button.append(image); figure.append(button, caption); watch(image, info.src);
    return figure;
  }
  globalThis.NexusMedia = {inlineImage, compactJSON, imageCard, prune, closePreview};
  if (typeof module !== 'undefined') module.exports = globalThis.NexusMedia;
})();
