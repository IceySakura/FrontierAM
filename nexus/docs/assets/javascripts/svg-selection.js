// Inject selection style into SVG objects
(function() {
  const selectionCSS = `
    ::selection {
      background-color: rgba(173, 216, 230, 0.6) !important;
      color: inherit !important;
    }
    ::-moz-selection {
      background-color: rgba(173, 216, 230, 0.6) !important;
      color: inherit !important;
    }
  `;

  const injected = new WeakSet();

  function injectStyleToSVG(obj) {
    if (injected.has(obj)) return true;

    try {
      const svgDoc = obj.contentDocument;
      if (!svgDoc) return false;

      const svg = svgDoc.querySelector('svg');
      if (!svg) return false;

      // Check if already injected (in case of page cache)
      if (svgDoc.querySelector('style[data-selection-style]')) {
        injected.add(obj);
        return true;
      }

      const style = svgDoc.createElementNS('http://www.w3.org/2000/svg', 'style');
      style.setAttribute('data-selection-style', 'true');
      style.textContent = selectionCSS;
      svg.insertBefore(style, svg.firstChild);
      injected.add(obj);
      return true;
    } catch (e) {
      // Cross-origin or other error
      return false;
    }
  }

  function processAllSVGs() {
    const objects = document.querySelectorAll('object[type="image/svg+xml"]');
    let allDone = true;

    objects.forEach(obj => {
      if (!injectStyleToSVG(obj)) {
        allDone = false;
        // Add load listener as fallback
        obj.addEventListener('load', () => injectStyleToSVG(obj), { once: true });
      }
    });

    return allDone;
  }

  // Poll until all SVGs are processed (handles async loading)
  let attempts = 0;
  const maxAttempts = 50; // 5 seconds max

  function poll() {
    if (processAllSVGs() || ++attempts >= maxAttempts) return;
    setTimeout(poll, 100);
  }

  // Start polling
  poll();

  // Also observe for dynamically added SVGs
  const observer = new MutationObserver(() => {
    attempts = 0;
    poll();
  });

  observer.observe(document.body, { childList: true, subtree: true });

  // Re-run on MkDocs navigation
  if (typeof document$ !== 'undefined') {
    document$.subscribe(() => {
      attempts = 0;
      poll();
    });
  }
})();
