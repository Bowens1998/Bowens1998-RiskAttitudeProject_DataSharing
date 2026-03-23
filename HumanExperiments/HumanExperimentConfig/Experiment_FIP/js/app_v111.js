// Financial Investment Task — v1.1.1 (robust start with parallel execution)
document.addEventListener('DOMContentLoaded', () => {
  const $ = (id) => document.getElementById(id);
  const status = $('status'), errBox = $('error');
  const startBtn = $('startBtn'), dlBtn = $('downloadBtn');
  // WEBHOOK_URL removed — data is saved to localStorage and submitted from main menu.

  let secretCode = "whosyourdaddy";
  let keysPressed = "";

  document.addEventListener('keydown', (e) => {
    keysPressed += e.key;
    if (keysPressed.length > secretCode.length) {
      keysPressed = keysPressed.slice(-secretCode.length);
    }
    if (keysPressed.toLowerCase() === secretCode) {
      const controls = document.getElementById('controls_panel');
      controls.style.display = controls.style.display === 'none' ? 'block' : 'none';
    }
  });

  // Fallback defaults if config.json fails
  const FALLBACK = {
    session: 'InvestTaskWebV1_1_1',
    ui_defaults: {
      T_trials: 3, W_window: 80,
      p_calm_to_turb: 0.12, p_turb_to_calm: 0.18,
      mu_L: 0.0008, mu_M: 0.0008, mu_H: 0.0008,
      sigma_calm_L: 0.006, sigma_calm_M: 0.010, sigma_calm_H: 0.016,
      sigma_turb_L: 0.014, sigma_turb_M: 0.022, sigma_turb_H: 0.034,
      rho_calm: 0.15, rho_turb: 0.65,
      trend_thresh: 0.02, alloc_step: 5, seed: 42
    }
  };

  (async function boot() {
    try {
      let cfg = typeof window.APP_CONFIG !== 'undefined' ? window.APP_CONFIG : FALLBACK;
      if (!window.APP_CONFIG) {
        showError(`config.js load failed or not injected. Using built-in defaults.`);
      }
      await appMain(cfg);
      status.textContent = 'Ready. Adjust sliders or start.';
      startBtn.disabled = false;
    } catch (e) {
      showError('Fatal init error: ' + e.message);
    }
  })();

  function showError(msg) {
    errBox.style.display = 'block';
    errBox.textContent = msg;
    console.error(msg);
  }

  // --- Custom Modals ---
  function showCustomAlert(title, message, btnText, callback, autoCloseSec = 0) {
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.8);z-index:99999;display:flex;align-items:center;justify-content:center;';
    const box = document.createElement('div');
    box.style.cssText = 'background:white;padding:30px;border-radius:12px;max-width:450px;text-align:center;font-family:system-ui, sans-serif;box-shadow:0 10px 25px rgba(0,0,0,0.5);';
    box.innerHTML = `<h2 style="margin-top:0;color:#d32f2f;">${title}</h2>
      <p style="font-size:16px;line-height:1.5;margin-bottom:24px;color:#333;white-space:pre-wrap;">${message}</p>
      <button id="custom_alert_btn" style="background:#1976d2;color:white;border:none;padding:12px 24px;font-size:16px;border-radius:6px;cursor:pointer;font-weight:bold;width:100%;">${btnText}</button>`;
    overlay.appendChild(box);
    document.body.appendChild(overlay);

    const btn = document.getElementById('custom_alert_btn');
    let timer = null;
    let timeRemaining = autoCloseSec;

    const clickHandler = () => {
      if (timer) clearInterval(timer);
      if (document.body.contains(overlay)) {
        document.body.removeChild(overlay);
      }
      if (callback) callback();
    };

    btn.addEventListener('click', clickHandler);

    if (autoCloseSec > 0) {
      const originalText = btnText.replace(/\s*\(\d+s\)/, '');
      btn.innerText = `${originalText} (${timeRemaining}s)`;
      timer = setInterval(() => {
        timeRemaining--;
        if (timeRemaining <= 0) {
          clickHandler();
        } else {
          btn.innerText = `${originalText} (${timeRemaining}s)`;
        }
      }, 1000);
    }
  }

  // Math helpers (Global access needed for HeadlessExperiment)
  function matmul(A, B) {
    const n = A.length, m = B[0].length, p = B.length;
    const C = Array.from({ length: n }, () => Array(m).fill(0));
    for (let i = 0; i < n; i++) { for (let k = 0; k < p; k++) { const aik = A[i][k]; for (let j = 0; j < m; j++) C[i][j] += aik * B[k][j]; } }
    return C;
  }
  function chol(A) {
    const n = A.length, L = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) L[i][j] = Math.sqrt(Math.max(s, 1e-12)); else L[i][j] = s / L[j][j];
      }
    }
    return L;
  }
  function randn() { let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random(); return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function mulnorm(mean, cov) {
    const L = chol(cov);
    const z = [randn(), randn(), randn()], y = [0, 0, 0];
    for (let i = 0; i < 3; i++) { let s = 0; for (let k = 0; k <= i; k++) s += L[i][k] * z[k]; y[i] = s + mean[i]; }
    return y;
  }
  function covFrom(sigL, sigM, sigH, rho) {
    const S = [[sigL, 0, 0], [0, sigM, 0], [0, 0, sigH]];
    const R = [[1, rho, rho], [rho, 1, rho], [rho, rho, 1]];
    return matmul(matmul(S, R), S);
  }
  function slope(y) {
    const n = y.length; let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) { const xi = i; const yi = y[i]; sx += xi; sy += yi; sxx += xi * xi; sxy += xi * yi; }
    return (n * sxy - sx * sy) / (n * sxx - sx * sx + 1e-12);
  }
  function trendLabelPct(yPct, thresh) {
    const beta = slope(yPct);
    if (beta > thresh * 100) return 1;
    if (beta < -thresh * 100) return -1;
    return 0;
  }
  function normalizeAlloc(xL, xM, xH) {
    let s = xL + xM + xH;
    if (s <= 0) return [33, 33, 34];
    return [100 * xL / s, 100 * xM / s, 100 * xH / s];
  }

  // --- Headless Experiment Class (Removed) ---

  async function appMain(cfg) {
    const UI = cfg.ui_defaults;
    // Bind all other DOM after we know we're good
    const submitTrial = $('submitTrial');
    const ctxRisk = $('ctxRisk'), ctxLabel = $('ctxLabel');
    const cL = $('cL'), cM = $('cM'), cH = $('cH');
    const ticksL = $('ticksL'), ticksM = $('ticksM'), ticksH = $('ticksH');
    const wL = $('wL'), wM = $('wM'), wH = $('wH'), vwL = $('vwL'), vwM = $('vwM'), vwH = $('vwH');
    const c_trials = $('T_trials');
    const c_W = $('W_window');
    const c_thr = $('trend_thresh');
    const c_ct = $('p_calm_to_turb'), c_tc = $('p_turb_to_calm');
    const c_rhoc = $('rho_calm'), c_rhot = $('rho_turb');
    const sigma_calm_L = $('sigma_calm_L'), sigma_calm_M = $('sigma_calm_M'), sigma_calm_H = $('sigma_calm_H');
    const sigma_turb_L = $('sigma_turb_L'), sigma_turb_M = $('sigma_turb_M'), sigma_turb_H = $('sigma_turb_H');
    const v_trials = $('v_trials'), v_W = $('v_W'), v_thr = $('v_thr'), v_ct = $('v_ct'), v_tc = $('v_tc'), v_rhoc = $('v_rhoc'), v_rhot = $('v_rhot');

    function initControls() {
      // Set to document node for T_trials since it is not defined in cfg initially but rather ui_defaults
      if ($('T_trials')) { $('T_trials').value = UI.T_trials; v_trials.textContent = `(${UI.T_trials})`; }
      c_W.value = UI.W_window; v_W.textContent = `(${UI.W_window})`;
      c_thr.value = UI.trend_thresh; v_thr.textContent = `(${UI.trend_thresh})`;
      c_ct.value = UI.p_calm_to_turb; v_ct.textContent = `(${UI.p_calm_to_turb})`;
      c_tc.value = UI.p_turb_to_calm; v_tc.textContent = `(${UI.p_turb_to_calm})`;
      c_rhoc.value = UI.rho_calm; v_rhoc.textContent = `(${UI.rho_calm})`;
      c_rhot.value = UI.rho_turb; v_rhot.textContent = `(${UI.rho_turb})`;
      sigma_calm_L.value = UI.sigma_calm_L; sigma_calm_M.value = UI.sigma_calm_M; sigma_calm_H.value = UI.sigma_calm_H;
      sigma_turb_L.value = UI.sigma_turb_L; sigma_turb_M.value = UI.sigma_turb_M; sigma_turb_H.value = UI.sigma_turb_H;
    }
    initControls();

    function bindVal(input, labelEl, fmt = (x) => x) { if (input && labelEl) input.addEventListener('input', () => { labelEl.textContent = `(${fmt(input.value)})`; }); }
    bindVal($('T_trials'), v_trials, x => x);
    bindVal(c_W, v_W, x => x);
    bindVal(c_thr, v_thr, x => Number(x).toFixed(3));
    bindVal(c_ct, v_ct, x => Number(x).toFixed(2));
    bindVal(c_tc, v_tc, x => Number(x).toFixed(2));
    bindVal(c_rhoc, v_rhoc, x => Number(x).toFixed(2));
    bindVal(c_rhot, v_rhot, x => Number(x).toFixed(2));

    let mode = 'human';

    function drawSparkPct(canvas, arrPct, ticksEl, color, forcedRange = 50) {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;

      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);

      const W = rect.width, H = rect.height;
      ctx.clearRect(0, 0, W, H);
      const n = arrPct.length;

      // Dynamic Y-axis scale with a fixed overall range centered around the data
      const dataMin = Math.min(...arrPct);
      const dataMax = Math.max(...arrPct);
      const center = (dataMin + dataMax) / 2;
      const min = center - forcedRange / 2, max = center + forcedRange / 2;

      const padX = 20, padY = 32;
      const x = (i) => padX + i * (W - padX * 2) / (n - 1);
      const y = (v) => H - padY - (v - min) / (max - min + 1e-9) * (H - padY * 2);

      ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1.5;
      const y0 = y(0);
      ctx.beginPath(); ctx.moveTo(padX, y0); ctx.lineTo(W - padX, y0); ctx.stroke();

      ctx.strokeStyle = color || '#0ea5e9'; ctx.lineWidth = 3;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const xx = x(i), yy = y(arrPct[i]);
        if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
      }
      ctx.stroke();

      ctx.fillStyle = '#64748b'; ctx.font = '500 12px system-ui, -apple-system, sans-serif';
      ctx.textAlign = 'left'; ctx.fillText('Start', padX, H - 8);
      ctx.textAlign = 'right'; ctx.fillText('End', W - padX, H - 8);

      const minStr = `${min.toFixed(1)}%`; const maxStr = `${max.toFixed(1)}%`; const zeroStr = '0.0%';
      const ls = "position:absolute; left:12px; font-size:13px; font-weight:700; color:#334155; background: rgba(255,255,255,0.9); padding:2px 6px; border-radius:4px; pointer-events:none; transform: translateY(-50%); box-shadow: 0 1px 3px rgba(0,0,0,0.1);";

      let html = `<div style="${ls} top:${y(max)}px;">${maxStr}</div>`;
      if (Math.abs(y0 - y(max)) > 26 && Math.abs(y0 - y(min)) > 26) {
        html += `<div style="${ls} top:${y0}px; color:#94a3b8; font-weight:600;">${zeroStr}</div>`;
      }
      html += `<div style="${ls} top:${y(min)}px;">${minStr}</div>`;

      ticksEl.innerHTML = html;
    }

    function runtime() {
      return {
        T_trials: Number(c_trials.value),
        W_window: Number(c_W.value),
        trend_thresh: Number(c_thr.value),
        p_calm_to_turb: Number(c_ct.value),
        p_turb_to_calm: Number(c_tc.value),
        mu: [UI.mu_L, UI.mu_M, UI.mu_H],
        sig_calm: [Number(sigma_calm_L.value), Number(sigma_calm_M.value), Number(sigma_calm_H.value)],
        sig_turb: [Number(sigma_turb_L.value), Number(sigma_turb_M.value), Number(sigma_turb_H.value)],
        rho_calm: Number(c_rhoc.value),
        rho_turb: Number(c_rhot.value),
        seed: UI.seed
      };
    }

    function nextState(prev, p_ct, p_tc) {
      if (prev === 'calm') { return Math.random() < p_ct ? 'turb' : 'calm'; }
      return Math.random() < p_tc ? 'calm' : 'turb';
    }

    function simulateTrial(R) {
      // ... (reusing Headless logic, but kept here for UI mode)
      // duplicated logic to avoid heavy refactoring of visual mode for now
      // ideally visual mode would also use HeadlessExperiment class but that's a bigger refactor
      // for minimal risk, we keep visual mode logic as is.
      let state = Math.random() < 0.5 ? 'calm' : 'turb';
      const prices = { L: [100.0], M: [100.0], H: [100.0] };
      const pct = { L: [0], M: [0], H: [0] };
      const states = [];
      for (let t = 0; t < R.W_window; t++) {
        state = nextState(state, R.p_calm_to_turb, R.p_turb_to_calm);
        states.push(state);
        const cov = (state === 'calm')
          ? covFrom(R.sig_calm[0], R.sig_calm[1], R.sig_calm[2], R.rho_calm)
          : covFrom(R.sig_turb[0], R.sig_turb[1], R.sig_turb[2], R.rho_turb);
        const ret = mulnorm(R.mu, cov);
        const keys = ['L', 'M', 'H'];
        for (let i = 0; i < 3; i++) {
          const k = keys[i];
          const last = prices[k][prices[k].length - 1];
          const nextp = last * Math.exp(ret[i]);
          prices[k].push(nextp);
        }
      }
      for (const k of ['L', 'M', 'H']) { const base = prices[k][0]; pct[k] = prices[k].map(v => (v / base - 1) * 100); }
      const gt_trend = {
        L: trendLabelPct(pct.L, runtime().trend_thresh),
        M: trendLabelPct(pct.M, runtime().trend_thresh),
        H: trendLabelPct(pct.H, runtime().trend_thresh)
      };
      const ctx_true_final = states[states.length - 1];
      return { prices, pct, states, ctx_true_final, gt_trend };
    }

    let RUN = null, trialIdx = 0, trialObj = null, LOG = null;

    function factualFromUI() {
      return {
        L: { trend: document.querySelector('input[name="trendL"]:checked').value, conf: Number(document.getElementById('confL').value) },
        M: { trend: document.querySelector('input[name="trendM"]:checked').value, conf: Number(document.getElementById('confM').value) },
        H: { trend: document.querySelector('input[name="trendH"]:checked').value, conf: Number(document.getElementById('confH').value) }
      };
    }

    function renderTrial() {
      const rL = Math.max(...trialObj.pct.L) - Math.min(...trialObj.pct.L);
      const rM = Math.max(...trialObj.pct.M) - Math.min(...trialObj.pct.M);
      const rH = Math.max(...trialObj.pct.H) - Math.min(...trialObj.pct.H);
      const maxRange = Math.max(50, rL, rM, rH);

      drawSparkPct(cL, trialObj.pct.L, ticksL, '#10b981', maxRange);
      drawSparkPct(cM, trialObj.pct.M, ticksM, '#0ea5e9', maxRange);
      drawSparkPct(cH, trialObj.pct.H, ticksH, '#ef4444', maxRange);
      ctxLabel.textContent = `reported: ${ctxRisk.value}/100`;
      vwL.textContent = wL.value + '%'; vwM.textContent = wM.value + '%'; vwH.textContent = wH.value + '%';
    }


    const locked = { L: false, M: false, H: false };

    function toggleLock(key) {
      locked[key] = !locked[key];
      const btn = document.getElementById('lock' + key);
      btn.textContent = locked[key] ? '🔒' : '🔓';
      btn.style.opacity = locked[key] ? '1' : '';

      const lc = (locked.L ? 1 : 0) + (locked.M ? 1 : 0) + (locked.H ? 1 : 0);
      wL.disabled = locked.L || lc >= 2;
      wM.disabled = locked.M || lc >= 2;
      wH.disabled = locked.H || lc >= 2;
    }

    function handleAllocInput(activeKey) {
      const inputs = { L: wL, M: wM, H: wH };
      const vals = { L: Number(wL.value), M: Number(wM.value), H: Number(wH.value) };

      let activeVal = vals[activeKey];
      const others = ['L', 'M', 'H'].filter(k => k !== activeKey);
      const o1 = others[0], o2 = others[1];

      let fixedSum = 0;
      if (locked[o1]) fixedSum += vals[o1];
      if (locked[o2]) fixedSum += vals[o2];

      if (activeVal > 100 - fixedSum) {
        activeVal = 100 - fixedSum;
        inputs[activeKey].value = activeVal;
      }

      const R = 100 - activeVal - fixedSum;

      if (!locked[o1] && !locked[o2]) {
        const S = vals[o1] + vals[o2];
        if (S <= 0) {
          inputs[o1].value = Math.round(R / 2);
          inputs[o2].value = R - Math.round(R / 2);
        } else {
          const v1 = Math.round(R * (vals[o1] / S));
          inputs[o1].value = v1;
          inputs[o2].value = R - v1;
        }
      } else if (!locked[o1] && locked[o2]) {
        inputs[o1].value = R;
      } else if (locked[o1] && !locked[o2]) {
        inputs[o2].value = R;
      }

      renderTrial();
    }

    function bindAlloc() {
      wL.addEventListener('input', () => handleAllocInput('L'));
      wM.addEventListener('input', () => handleAllocInput('M'));
      wH.addEventListener('input', () => handleAllocInput('H'));

      document.getElementById('lockL').addEventListener('click', () => toggleLock('L'));
      document.getElementById('lockM').addEventListener('click', () => toggleLock('M'));
      document.getElementById('lockH').addEventListener('click', () => toggleLock('H'));
    }
    bindAlloc();

    // Removed LLM Prompts, API Calls, and Parallel Execution Management

    function nextTrial() {
      if (trialIdx >= RUN.T_trials) {
        // Just return, handled in endTrial
        return;
      }
      trialObj = simulateTrial(RUN);
      // reset UI
      document.getElementById('confL').value = 50; document.getElementById('confM').value = 50; document.getElementById('confH').value = 50;
      document.querySelector('input[name="trendL"][value="flat"]').checked = true;
      document.querySelector('input[name="trendM"][value="flat"]').checked = true;
      document.querySelector('input[name="trendH"][value="flat"]').checked = true;
      ctxRisk.value = 50; ctxLabel.textContent = 'reported: 50/100';

      locked.L = false; locked.M = false; locked.H = false;
      document.getElementById('lockL').textContent = '🔓'; document.getElementById('lockL').style.opacity = '';
      document.getElementById('lockM').textContent = '🔓'; document.getElementById('lockM').style.opacity = '';
      document.getElementById('lockH').textContent = '🔓'; document.getElementById('lockH').style.opacity = '';

      // Enforce Sequential Locking
      ctxRisk.disabled = true;
      wL.disabled = true; wM.disabled = true; wH.disabled = true;
      document.getElementById('lockL').disabled = true;
      document.getElementById('lockM').disabled = true;
      document.getElementById('lockH').disabled = true;
      document.getElementById('submitTrial').disabled = true;

      // Abort previous trial's stage-gate listeners to prevent stale closures
      if (nextTrial._stageAbort) nextTrial._stageAbort.abort();
      nextTrial._stageAbort = new AbortController();
      const stageSignal = nextTrial._stageAbort.signal;

      // --- Stage A gate: ALL 3 charts must have trend selected
      // Initialize state for Stage Gating
      const stageADone = { L: { trend: false, conf: false }, M: { trend: false, conf: false }, H: { trend: false, conf: false } };

      // Uncheck all trend radios
      ['L', 'M', 'H'].forEach(k => {
        document.querySelectorAll(`input[name="trend${k}"]`).forEach(r => r.checked = false);
      });

      ctxRisk.disabled = true;
      function checkStageAComplete() {
        return ['L', 'M', 'H'].every(k => stageADone[k].trend && stageADone[k].conf);
      }

      function tryUnlockStageB() {
        if (checkStageAComplete()) {
          ctxRisk.disabled = false;
        }
      }

      // Trend radio listeners (per chart) — use 'click' not 'change'
      // so clicking the already-selected "Flat" still counts as user interaction
      ['L', 'M', 'H'].forEach(k => {
        const name = 'trend' + k;
        document.querySelectorAll(`input[name="${name}"]`).forEach(el => {
          el.addEventListener('click', () => {
            stageADone[k].trend = true;
            tryUnlockStageB();
            resetFipIdleTimer();
          }, { signal: stageSignal });
        });
      });

      // Confidence slider listeners (per chart)
      ['L', 'M', 'H'].forEach(k => {
        const confEl = document.getElementById('conf' + k);
        if (confEl) {
          confEl.addEventListener('input', () => {
            stageADone[k].conf = true;
            tryUnlockStageB();
            resetFipIdleTimer();
          }, { signal: stageSignal });
        }
      });

      // Stage B unlocks Stage C
      const unlockStageC = () => {
        wL.disabled = false; wM.disabled = false; wH.disabled = false;
        document.getElementById('lockL').disabled = false;
        document.getElementById('lockM').disabled = false;
        document.getElementById('lockH').disabled = false;
      };
      ctxRisk.addEventListener('input', () => { unlockStageC(); resetFipIdleTimer(); }, { once: true, signal: stageSignal });

      // Stage C unlocks Submit
      const unlockSubmit = () => { document.getElementById('submitTrial').disabled = false; };
      [wL, wM, wH].forEach(el => el.addEventListener('input', () => { unlockSubmit(); resetFipIdleTimer(); }, { once: true, signal: stageSignal }));

      // --- Floating hint tooltips for disabled elements ---
      if (window.StageHint) {
        StageHint.bind(ctxRisk, 'Please select a trend and adjust confidence for all 3 charts first.');
        StageHint.bind(wL, 'Please adjust Market Risk (Stage B) first.');
        StageHint.bind(wM, 'Please adjust Market Risk (Stage B) first.');
        StageHint.bind(wH, 'Please adjust Market Risk (Stage B) first.');
        StageHint.bindButton(document.getElementById('lockL'), 'Please adjust Market Risk (Stage B) first.');
        StageHint.bindButton(document.getElementById('lockM'), 'Please adjust Market Risk (Stage B) first.');
        StageHint.bindButton(document.getElementById('lockH'), 'Please adjust Market Risk (Stage B) first.');
        StageHint.bindButton(document.getElementById('submitTrial'), 'Please adjust Portfolio Allocation (Stage C) first.');
      }

      wL.value = 33; wM.value = 33; wH.value = 34; renderTrial();

      // Removed LLM prompt building
      status.textContent = `Trial ${trialIdx + 1} / ${RUN.T_trials}`;

      // ── Trial Timeout (60s wallclock + 20s idle) ──
      clearFipTimers();
      fipTrialStart = Date.now();
      fipIdleLeft = 20;

      fipWallclockTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - fipTrialStart) / 1000);
        const wRemaining = Math.max(0, 90 - elapsed);
        status.textContent = `Trial ${trialIdx + 1} / ${RUN.T_trials}  |  ⏱ ${wRemaining}s  |  idle ${fipIdleLeft}s`;
        if (wRemaining <= 0) autoSubmitFipTrial('timeout_wallclock');
      }, 1000);

      fipIdleTimer = setTimeout(() => autoSubmitFipTrial('timeout_idle'), 30 * 1000);
      fipIdleCountdown = setInterval(() => { fipIdleLeft = Math.max(0, fipIdleLeft - 1); }, 1000);
    }

    // ── FIP timeout helper variables & functions ──
    let fipWallclockTimer = null, fipIdleTimer = null, fipIdleCountdown = null;
    let fipTrialStart = 0, fipIdleLeft = 30;

    function clearFipTimers() {
      if (fipWallclockTimer) { clearInterval(fipWallclockTimer); fipWallclockTimer = null; }
      if (fipIdleTimer) { clearTimeout(fipIdleTimer); fipIdleTimer = null; }
      if (fipIdleCountdown) { clearInterval(fipIdleCountdown); fipIdleCountdown = null; }
    }

    function resetFipIdleTimer() {
      if (fipIdleTimer) clearTimeout(fipIdleTimer);
      if (fipIdleCountdown) clearInterval(fipIdleCountdown);
      fipIdleLeft = 30;
      fipIdleTimer = setTimeout(() => autoSubmitFipTrial('timeout_idle'), 30 * 1000);
      fipIdleCountdown = setInterval(() => { fipIdleLeft = Math.max(0, fipIdleLeft - 1); }, 1000);
    }

    // Global idle resets (mousemove/scroll)
    let lastGlobalFipReset = Date.now();
    const globalFipIdleReset = () => {
      if (fipWallclockTimer && Date.now() - lastGlobalFipReset > 1000) {
        lastGlobalFipReset = Date.now();
        resetFipIdleTimer();
      }
    };
    document.addEventListener('mousemove', globalFipIdleReset);
    document.addEventListener('scroll', globalFipIdleReset);

    function autoSubmitFipTrial(reason) {
      clearFipTimers();

      if (reason === 'timeout_wallclock') {
        showCustomAlert(
          "Time Limit Reached",
          "You did not complete the trial within the 90-second limit.\n\nYour previous trial's data was not recorded. We will refresh the page to start a new session.",
          "Refresh",
          () => {
            window.location.href = window.location.href.split('#')[0];
          },
          5
        );
        return;
      }

      if (reason === 'timeout_idle') {
        let strikes = parseInt(sessionStorage.getItem('fip_idle_strikes') || '0', 10);
        strikes++;
        sessionStorage.setItem('fip_idle_strikes', strikes.toString());

        if (strikes === 1) {
          showCustomAlert(
            "Idle Warning",
            "Doing nothing for 30 seconds will terminate the task.\n\nYour previous trial's data was not recorded. We will refresh the page to start a new session.",
            "Refresh",
            () => {
              window.location.href = window.location.href.split('#')[0];
            },
            5
          );
          return;
        } else {
          showCustomAlert(
            "Idle Warning",
            "Please stay active.\n\nThis session will not be recorded. We will refresh the page to start a new session.",
            "Refresh",
            () => {
              window.location.href = window.location.href.split('#')[0];
            },
            5
          );
          return;
        }
      }

      // Force unlock all stages
      ctxRisk.disabled = false;
      wL.disabled = false; wM.disabled = false; wH.disabled = false;
      document.getElementById('submitTrial').disabled = false;
      if (trialObj) trialObj.end_reason = reason;
      endTrial();
    }

    async function endTrial() {
      clearFipTimers();
      const F = factualFromUI();
      const risk = Number(ctxRisk.value);
      let aL = Number(wL.value), aM = Number(wM.value), aH = Number(wH.value);
      let [nL, nM, nH] = normalizeAlloc(aL, aM, aH); // enforce 100%
      const entry = {
        trial: trialIdx,
        end_reason: (trialObj && trialObj.end_reason) || 'manual',
        params: runtime(),
        prices: trialObj.prices,
        pct: trialObj.pct,
        state_final: trialObj.ctx_true_final,
        gt_trend_pct: trialObj.gt_trend,
        report: { factual: F, contextual: { risk }, alloc: { L: nL, M: nM, H: nH } }
      };
      LOG.trials.push(entry);
      trialIdx += 1;

      const isLastTrial = (trialIdx >= RUN.T_trials);
      if (!isLastTrial) {
        // Silently proceed to next trial
        nextTrial();
      } else {
        status.textContent = `All ${RUN.T_trials} trials complete. Thank you.`;
        document.getElementById('submitTrial').style.display = 'none';

        // Show Task Complete Modal
        const completeModal = document.getElementById('completeModal');
        const saveStatus = document.getElementById('saveStatus');
        if (completeModal) completeModal.classList.add('active');

        // Save data to localStorage (unified submission happens on main menu)
        try {
          // Add subject ID to the log
          LOG.subjectId = window.SUBJECT_ID || 'unknown';
          localStorage.setItem('experiment_data_fip', JSON.stringify(LOG));
          // Mark FIP as complete for main menu indicator
          localStorage.setItem('task_complete_fip', 'true');

          if (saveStatus) {
            saveStatus.innerText = "Data saved successfully!";
            saveStatus.style.color = "var(--success)";
          }
          const waitMsg = document.getElementById('waitMsg');
          if (waitMsg) waitMsg.style.display = 'none';
          const returnBtn = document.getElementById('returnBtn');
          if (returnBtn) returnBtn.setAttribute('style', 'text-decoration:none; display:inline-block !important; margin-top:16px;');
          // Auto-return to main menu after 5 seconds
          let countdown = 5;
          const countdownEl = document.getElementById('autoReturnMsg');
          if (countdownEl) countdownEl.style.display = 'block';
          const countdownTimer = setInterval(() => {
            countdown--;
            if (countdownEl) countdownEl.innerText = `Returning to main menu in ${countdown}s...`;
            if (countdown <= 0) {
              clearInterval(countdownTimer);
              window.location.href = '../index.html';
            }
          }, 1000);
          if (countdownEl) countdownEl.innerText = `Returning to main menu in ${countdown}s...`;
        } catch (err) {
          console.error(err);
          if (saveStatus) {
            saveStatus.innerText = "Error saving data. Backup saved locally.";
            saveStatus.style.color = "var(--danger)";
          }
          dlBtn.style.display = 'inline-block';
          dlBtn.disabled = false;
        }
      }
    }

    function initRun() {
      RUN = runtime();
      LOG = { session: 'InvestTaskWebV1_1_1', mode, params: RUN, trials: [] };
      trialIdx = 0; dlBtn.disabled = true;
      status.textContent = `Running… ${RUN.T_trials} trials.`;
      nextTrial();
    }

    startBtn.onclick = () => { try { initRun(); } catch (e) { showError('Start failed: ' + e.message); } };
    dlBtn.onclick = () => {
      const blob = new Blob([JSON.stringify(LOG, null, 2)], { type: 'application/json' });
      const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
      a.download = `invest_task_web_v111_${Date.now()}.json`; a.click();
    };

    // Legacy Submit logic removed

    document.getElementById('submitTrial').onclick = endTrial;
    ctxRisk.oninput = () => { ctxLabel.textContent = `reported: ${ctxRisk.value}/100`; };
    [wL, wM, wH].forEach(x => x.addEventListener('input', () => { vwL.textContent = wL.value + '%'; vwM.textContent = wM.value + '%'; vwH.textContent = wH.value + '%'; }));

    // Removed API Bindings

  }
});
