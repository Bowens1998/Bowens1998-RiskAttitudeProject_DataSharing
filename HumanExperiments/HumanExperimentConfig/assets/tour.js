// tour.js

class TourGuide {
    constructor(steps, onComplete = null, options = {}) {
        this.steps = steps;
        this.currentStep = 0;
        this.overlay = null;
        this.tooltip = null;
        this.targetElement = null;
        this.timer = null;
        this.onComplete = onComplete; // Callback after tour finishes
        this.autoAdvanceSec = options.autoAdvanceSec || 0; // 0 = disabled
        this._autoAdvanceTimer = null;
    }

    start() {
        this.createOverlay();
        this.createTooltip();
        this.bindEvents();

        // Slight delay to allow DOM render
        setTimeout(() => {
            this.overlay.style.opacity = '1';
            this.showStep();
        }, 50);
    }

    createOverlay() {
        this.overlay = document.createElement('div');
        this.overlay.className = 'tour-overlay';
        document.body.appendChild(this.overlay);
    }

    createTooltip() {
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'tour-tooltip';
        document.body.appendChild(this.tooltip);
    }

    showStep() {
        // Clear any previous step's countdown timer to prevent stale timers from running
        if (this._stepCountdown) {
            clearInterval(this._stepCountdown);
            this._stepCountdown = null;
        }

        // Clear and restart auto-advance timer
        if (this._autoAdvanceTimer) {
            clearTimeout(this._autoAdvanceTimer);
            this._autoAdvanceTimer = null;
        }

        const step = this.steps[this.currentStep];

        if (this.autoAdvanceSec > 0 && !step.skipDelay && !step.hideNext) {
            this._autoAdvanceTimer = setTimeout(() => {
                this.nextStep();
            }, this.autoAdvanceSec * 1000);
        }

        if (this.targetElement) {
            this.targetElement.classList.remove('tour-highlight');
            if (this.targetElement.dataset.tourOriginalDisplay !== undefined) {
                this.targetElement.style.display = this.targetElement.dataset.tourOriginalDisplay;
            }
        }

        this.targetElement = step.target ? document.querySelector(step.target) : null;

        if (this.targetElement) {
            const computedStyle = window.getComputedStyle(this.targetElement);
            if (computedStyle.display === 'none') {
                this.targetElement.dataset.tourOriginalDisplay = this.targetElement.style.display;
                this.targetElement.style.display = 'block';
            }
            this.targetElement.classList.add('tour-highlight');
            this.targetElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        if (typeof step.onShow === 'function') {
            step.onShow(this.targetElement, this.tooltip);
        }

        this.renderTooltipContent(step);
        this.positionTooltip(step.placement || 'bottom');

        requestAnimationFrame(() => {
            this.tooltip.classList.add('show');
        });

        const nextBtn = this.tooltip.querySelector('.tour-next-btn');
        if (nextBtn) {
            if (step.hideNext) {
                nextBtn.style.display = 'none';
            } else {
                nextBtn.style.display = 'inline-block';
                nextBtn.innerText = this.currentStep === this.steps.length - 1 ? 'Start Experiment' : 'Next Step';

                // Skip global countdown for steps that manage their own locking (diseaseExamplesBox)
                if (!step.noDelay && !step.skipDelay) {
                    const finalLabel = this.currentStep === this.steps.length - 1 ? 'Start Experiment' : 'Next Step';
                    let remaining = 3;
                    nextBtn.disabled = true;
                    nextBtn.style.opacity = '0.5';
                    nextBtn.style.cursor = 'not-allowed';
                    nextBtn.innerText = `${finalLabel} (${remaining}s)`;
                    this._stepCountdown = setInterval(() => {
                        remaining--;
                        if (remaining <= 0) {
                            clearInterval(this._stepCountdown);
                            this._stepCountdown = null;
                            nextBtn.disabled = false;
                            nextBtn.style.opacity = '1';
                            nextBtn.style.cursor = 'pointer';
                            nextBtn.innerText = finalLabel;
                        } else {
                            nextBtn.innerText = `${finalLabel} (${remaining}s)`;
                        }
                    }, 1000);
                }
            }
        }
    }

    renderTooltipContent(step) {
        let dotsHtml = '';
        for (let i = 0; i < this.steps.length; i++) {
            dotsHtml += `<div class="tour-dot ${i === this.currentStep ? 'active' : ''}"></div>`;
        }

        let mediaHtml = '';
        if (step.media) {
            mediaHtml = `
            <div class="tour-media">
                <img src="${step.media}" alt="Tutorial Demonstration">
            </div>
        `;
        }

        this.tooltip.innerHTML = `
      <h3>${step.title}</h3>
      <div class="tour-text-content" style="margin-bottom:12px;">${step.content}</div>
      ${mediaHtml}
      
      <div class="tour-footer">
        <div class="tour-dots">
            ${dotsHtml}
        </div>
        <button class="tour-next-btn" id="tourNextBtn">Next</button>
      </div>
    `;

        document.getElementById('tourNextBtn').addEventListener('click', () => {
            this.nextStep();
        });
    }

    positionTooltip(placement) {
        if (!this.targetElement) {
            // Center on screen if no target
            this.tooltip.style.left = '50%';
            this.tooltip.style.top = '50%';
            this.tooltip.style.transform = 'translate(-50%, -50%) scale(1)';
            return;
        }

        // Position relative to target
        const targetRect = this.targetElement.getBoundingClientRect();
        const tooltipRect = this.tooltip.getBoundingClientRect();

        // Add some padding from the target
        const gap = 30; // Increased gap to prevent overlap

        let top, left;

        // Incorporate scroll offset for absolute positioning
        const scrollY = window.scrollY;
        const scrollX = window.scrollX;

        switch (placement) {
            case 'top':
                top = targetRect.top + scrollY - tooltipRect.height - gap;
                left = targetRect.left + scrollX + (targetRect.width / 2) - (tooltipRect.width / 2);
                break;
            case 'bottom':
                top = targetRect.bottom + scrollY + gap;
                left = targetRect.left + scrollX + (targetRect.width / 2) - (tooltipRect.width / 2);
                break;
            case 'left':
                top = targetRect.top + scrollY + (targetRect.height / 2) - (tooltipRect.height / 2);
                left = targetRect.left + scrollX - tooltipRect.width - gap;
                break;
            case 'right':
                top = targetRect.top + scrollY + (targetRect.height / 2) - (tooltipRect.height / 2);
                left = targetRect.right + scrollX + gap;
                break;
            default:
                // Default center fall back
                top = targetRect.bottom + scrollY + gap;
                left = targetRect.left + scrollX + (targetRect.width / 2) - (tooltipRect.width / 2);
        }

        // Simple bounds checking (keep within document body, roughly)
        if (left < 10) left = 10;
        if (left + tooltipRect.width > document.documentElement.scrollWidth - 10) left = document.documentElement.scrollWidth - tooltipRect.width - 10;
        if (top < 10) top = 10; // If too high, basic check for now.

        // For very long pages, position fixed is relative to viewport anyway.
        this.tooltip.style.left = `${left}px`;
        this.tooltip.style.top = `${top}px`;
        // Remove the center transform logic if we are setting absolute coordinates
        this.tooltip.style.transform = 'translate(0, 0) scale(1)';
    }

    nextStep() {
        if (this._isTransitioning) return;
        if (this._autoAdvanceTimer) { clearTimeout(this._autoAdvanceTimer); this._autoAdvanceTimer = null; }
        this._isTransitioning = true;

        const step = this.steps[this.currentStep];
        if (typeof step.onNext === 'function') {
            step.onNext(this.targetElement);
        }

        this.tooltip.classList.remove('show');
        setTimeout(() => {
            this.currentStep++;
            this._isTransitioning = false; // Release lock
            if (this.currentStep >= this.steps.length) {
                this.finish();
            } else {
                this.showStep();
            }
        }, 300); // Wait for fade out
    }

    goToStep(index) {
        if (index >= 0 && index < this.steps.length) {
            this.tooltip.classList.remove('show');
            setTimeout(() => {
                this.currentStep = index;
                this.showStep();
            }, 300);
        }
    }

    finish() {
        if (this._isFinished) return;
        this._isFinished = true;
        this._isTransitioning = false;
        clearInterval(this.timer);
        if (this._autoAdvanceTimer) { clearTimeout(this._autoAdvanceTimer); this._autoAdvanceTimer = null; }
        this.overlay.style.opacity = '0';
        this.tooltip.style.opacity = '0';

        if (this.targetElement) {
            this.targetElement.classList.remove('tour-highlight');
            if (this.targetElement.dataset.tourOriginalDisplay !== undefined) {
                this.targetElement.style.display = this.targetElement.dataset.tourOriginalDisplay;
            }
        }

        setTimeout(() => {
            if (this.overlay && this.overlay.parentNode) this.overlay.parentNode.removeChild(this.overlay);
            if (this.tooltip && this.tooltip.parentNode) this.tooltip.parentNode.removeChild(this.tooltip);

            if (this.onComplete) {
                this.onComplete();
            }
        }, 300);

        // Cleanup global listeners if we added them
        window.removeEventListener('resize', this.boundReposition);
        window.removeEventListener('scroll', this.boundReposition);
    }

    bindEvents() {
        this.boundReposition = () => {
            if (this.tooltip && this.tooltip.classList.contains('show')) {
                this.positionTooltip(this.steps[this.currentStep].placement || 'bottom');
            }
        };
        window.addEventListener('resize', this.boundReposition);
        window.addEventListener('scroll', this.boundReposition, true);
    }
}
