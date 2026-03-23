/**
 * Shared floating hint utility for all experiments.
 * Shows a small tooltip near the mouse cursor when the user clicks on
 * a disabled / locked element, then fades out after 3 seconds.
 *
 * Usage:
 *   1. Include this script via <script src="../assets/stage-hint.js"></script>
 *   2. Call StageHint.bind(element, message) for every disabled target.
 *   3. Call StageHint.bindButton(button, message) for disabled buttons.
 *   4. Safe to call multiple times — updates message if already bound.
 */
window.StageHint = (() => {
    // Inject CSS once
    const style = document.createElement('style');
    style.textContent = `
    .stage-hint-float {
      position: fixed;
      z-index: 100000;
      background: #1e293b;
      color: #f1f5f9;
      font-size: 13px;
      font-family: 'Inter', system-ui, sans-serif;
      padding: 8px 14px;
      border-radius: 8px;
      box-shadow: 0 4px 14px rgba(0,0,0,0.25);
      pointer-events: none;
      max-width: 320px;
      white-space: normal;
      opacity: 1;
      transition: opacity 0.6s ease;
    }
    .stage-hint-float.fade-out {
      opacity: 0;
    }
  `;
    document.head.appendChild(style);

    let activeHint = null;

    function show(x, y, message) {
        if (activeHint) {
            activeHint.remove();
            activeHint = null;
        }

        const el = document.createElement('div');
        el.className = 'stage-hint-float';
        el.textContent = message;
        document.body.appendChild(el);
        activeHint = el;

        const pad = 12;
        let left = x + pad;
        let top = y + pad;
        requestAnimationFrame(() => {
            const rect = el.getBoundingClientRect();
            if (left + rect.width > window.innerWidth - 8) left = x - rect.width - pad;
            if (top + rect.height > window.innerHeight - 8) top = y - rect.height - pad;
            el.style.left = left + 'px';
            el.style.top = top + 'px';
        });
        el.style.left = left + 'px';
        el.style.top = top + 'px';

        setTimeout(() => el.classList.add('fade-out'), 2400);
        setTimeout(() => { el.remove(); if (activeHint === el) activeHint = null; }, 3000);
    }

    /**
     * Bind a click interceptor to an input/slider that shows a hint when disabled.
     * Uses a pointerdown listener on the parent container instead of wrapping,
     * which avoids breaking flex/grid layouts.
     */
    function bind(element, message) {
        if (!element) return;

        // Store message on the element so it can be updated
        element._stageHintMsg = message;

        if (element.dataset.stageHintBound) return; // Already bound
        element.dataset.stageHintBound = '1';

        // Listen on the parent for pointerdown — if the element is disabled,
        // the click won't reach the element itself but WILL reach the parent.
        const parent = element.closest('.card, .panel, fieldset, .slider-wrap, .alloc') || element.parentElement;
        if (!parent) return;

        parent.addEventListener('pointerdown', (e) => {
            // Check if the click is over the disabled element's area
            if (!element.disabled) return;
            const rect = element.getBoundingClientRect();
            const inBounds = e.clientX >= rect.left && e.clientX <= rect.right &&
                e.clientY >= rect.top && e.clientY <= rect.bottom;
            if (inBounds) {
                show(e.clientX, e.clientY, element._stageHintMsg);
            }
        }, true);
    }

    /**
     * Bind a click interceptor to a button that shows a hint when disabled.
     * Uses a wrapper span approach since disabled buttons don't fire events.
     */
    function bindButton(button, message) {
        if (!button) return;

        // Store message so it can be updated
        button._stageHintMsg = message;

        if (button.dataset.stageHintBound) return; // Already bound
        button.dataset.stageHintBound = '1';

        // Wrap button in a span that can receive clicks even when button is disabled
        const wrapper = document.createElement('span');
        wrapper.style.cssText = 'display:inline-block; position:relative;';
        button.parentNode.insertBefore(wrapper, button);
        wrapper.appendChild(button);

        wrapper.addEventListener('click', (e) => {
            if (button.disabled) {
                e.preventDefault();
                e.stopPropagation();
                show(e.clientX, e.clientY, button._stageHintMsg);
            }
        }, true);
    }

    /**
     * Show hint programmatically at specific coordinates.
     */
    function showAt(x, y, message) {
        show(x, y, message);
    }

    return { bind, bindButton, showAt };
})();
