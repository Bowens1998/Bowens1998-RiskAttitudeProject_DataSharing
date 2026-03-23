// DSB v6 with Live Controls + OpenAI API Integration
(async function () {
  const defaults = window.APP_CONFIG;
  let config = JSON.parse(JSON.stringify(defaults)); // shallow clone for mutation by UI

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

  // Cheat code logic
  let keysPressed = "";
  const secretCode = "whosyourdaddy";
  document.addEventListener('keydown', (e) => {
    keysPressed += e.key;
    if (keysPressed.length > secretCode.length) {
      keysPressed = keysPressed.slice(-secretCode.length);
    }
    if (keysPressed.toLowerCase() === secretCode) {
      const controls = document.getElementById('controls');
      controls.style.display = controls.style.display === 'none' ? 'grid' : 'none';
    }
  });

  // --- UI bindings ---
  const EL = (id) => document.getElementById(id);
  const getControls = () => {
    const drift_range = parseInt(EL('ctl_drift_range').value, 10);
    const drift_values = Array.from({ length: drift_range * 2 + 1 }, (_, i) => i - drift_range);

    return {
      max_steps: parseInt(EL('ctl_max_steps').value, 10),
      cols: parseInt(EL('ctl_cols').value, 10),
      drain_step: parseFloat(EL('ctl_drain_step').value),
      drain_coll: parseFloat(EL('ctl_drain_coll').value),
      urgency_extra: parseFloat(EL('ctl_urgency').value),
      vol_low: parseFloat(EL('ctl_vol_low').value),
      vol_high: parseFloat(EL('ctl_vol_high').value),
      gust_rate: parseFloat(EL('ctl_gust_rate').value),
      bias: parseFloat(EL('ctl_bias').value),
      dense: parseFloat(EL('ctl_dense').value),
      sparse: parseFloat(EL('ctl_sparse').value),
      corr_keep: parseFloat(EL('ctl_corr_keep').value),
      drift_values,
      drift_probability: parseFloat(EL('ctl_drift_prob').value),
      trials_per_block: parseInt(EL('ctl_trials').value, 10) || 5
    };
  };

  // --- Elements ---
  const gridEl = document.getElementById('grid');
  const hud = document.getElementById('hud');
  const status = document.getElementById('status');
  const startBtn = document.getElementById('startBtn');
  const dlBtn = document.getElementById('downloadBtn');
  const submitBtn = document.getElementById('submitBtn');
  const beliefSlider = document.getElementById('beliefSlider');
  const beliefVal = document.getElementById('beliefVal');

  const surveyModal = document.getElementById('surveyModal');
  const okSurvey = document.getElementById('okSurvey');
  const skipSurvey = document.getElementById('skipSurvey');

  const completeModal = document.getElementById('completeModal');
  const saveStatus = document.getElementById('saveStatus');

  // Removed analyze button logic

  let mode = 'human';

  let runLog = null;

  function applyUI() {
    const u = getControls();
    config.trial.max_steps = u.max_steps;
    config.grid.cols = u.cols;
    config.battery.drain_per_step = u.drain_step;
    config.battery.drain_collision_penalty = u.drain_coll;
    config.battery.urgency_extra = u.urgency_extra; // custom field
    config._vol_low = u.vol_low;
    config._vol_high = u.vol_high;
    config._gust_rate = u.gust_rate;
    config._bias = u.bias;
    config._dense = u.dense;
    config._sparse = u.sparse;
    config._corr_keep = u.corr_keep;
    config.drift_values = u.drift_values;
    config.drift_probability = u.drift_probability;

    // reflow grid
    gridEl.style.gridTemplateRows = `repeat(${config.grid.rows}, ${config.grid.cell}px)`;
    gridEl.style.gridTemplateColumns = `repeat(${config.grid.cols}, ${config.grid.cell}px)`;
  }

  function rand(arr) { return arr[Math.floor(Math.random() * arr.length)]; }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function newMap(density) {
    const rows = config.grid.rows, cols = config.grid.cols;
    const startR = Math.floor(rows / 2), startC = 0;
    const goalR = Math.floor(rows / 2), goalC = cols - 1;

    // BFS: check if start can reach goal
    function hasPath(walls) {
      const visited = new Set();
      const queue = [[startR, startC]];
      visited.add(`${startR},${startC}`);
      while (queue.length > 0) {
        const [r, c] = queue.shift();
        if (r === goalR && c === goalC) return true;
        for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
          const nr = r + dr, nc = c + dc;
          if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
            const key = `${nr},${nc}`;
            if (!walls.has(key) && !visited.has(key)) {
              visited.add(key);
              queue.push([nr, nc]);
            }
          }
        }
      }
      return false;
    }

    function generate() {
      const p = density === 'dense' ? config._dense : config._sparse;
      const walls = new Set();
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          if ((r === startR && c === startC) || (r === goalR && c === goalC)) continue;
          if (Math.random() < p) walls.add(`${r},${c}`);
        }
      }
      // Clear some obstacles on the middle corridor
      for (let c = 0; c < cols; c++) {
        if (Math.random() < config._corr_keep) continue;
        walls.delete(`${startR},${c}`);
      }

      // Ensure no column is completely blocked
      for (let c = 1; c < cols - 1; c++) {
        let blockedCount = 0;
        for (let r = 0; r < rows; r++) {
          if (walls.has(`${r},${c}`)) blockedCount++;
        }
        if (blockedCount === rows) {
          const unblockRow = Math.floor(Math.random() * rows);
          walls.delete(`${unblockRow},${c}`);
        }
      }

      // Ensure start and goal are not confined (can reach ≥15 open tiles each)
      function ensureReachable(sr, sc) {
        const visited = new Set();
        const queue = [[sr, sc]];
        visited.add(`${sr},${sc}`);
        let openCount = 0;
        while (queue.length > 0) {
          const [currR, currC] = queue.shift();
          openCount++;
          if (openCount >= 15) return true;
          for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
            const nr = currR + dr, nc = currC + dc;
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
              const key = `${nr},${nc}`;
              if (!walls.has(key) && !visited.has(key)) {
                visited.add(key);
                queue.push([nr, nc]);
              }
            }
          }
        }
        return false;
      }

      let safety = 100;
      while (!ensureReachable(startR, startC) && safety-- > 0) {
        let r = clamp(startR - 2 + Math.floor(Math.random() * 5), 0, rows - 1);
        let c = Math.max(0, startC + Math.floor(Math.random() * 4));
        walls.delete(`${r},${c}`);
      }
      safety = 100;
      while (!ensureReachable(goalR, goalC) && safety-- > 0) {
        let r = clamp(goalR - 2 + Math.floor(Math.random() * 5), 0, rows - 1);
        let c = clamp(goalC - 3 + Math.floor(Math.random() * 4), 0, cols - 1);
        walls.delete(`${r},${c}`);
      }

      return walls;
    }

    // Generate maps until we get one with a valid path (up to 10 attempts)
    for (let attempt = 0; attempt < 10; attempt++) {
      const walls = generate();
      if (hasPath(walls)) return walls;

      // No path — punch holes along the middle corridor and retry check
      for (let c = 0; c < cols; c++) {
        walls.delete(`${startR},${c}`);
        if (Math.random() < 0.3 && startR > 0) walls.delete(`${startR - 1},${c}`);
        if (Math.random() < 0.3 && startR < rows - 1) walls.delete(`${startR + 1},${c}`);
      }
      if (hasPath(walls)) return walls;
    }

    // Ultimate fallback: clear the entire middle row
    const walls = generate();
    for (let c = 0; c < cols; c++) {
      walls.delete(`${startR},${c}`);
    }
    return walls;
  }

  function driftGenerators(volatility) {
    const rows = config.grid.rows;
    const gens = [];
    const base = (volatility === 'high') ? config._vol_high : config._vol_low;
    for (let r = 0; r < rows; r++) {
      gens.push((function* () {
        let d = rand(config.drift_values);
        while (true) {
          yield d;
          if (Math.random() < base) d = rand(config.drift_values);
        }
      })());
    }
    return gens;
  }

  function render(agent, walls, battery) {
    const rows = config.grid.rows, cols = config.grid.cols, cell = config.grid.cell;
    gridEl.innerHTML = '';
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const div = document.createElement('div');
        div.className = 'cell';
        const key = `${r},${c}`;
        if (walls.has(key)) div.classList.add('wall');
        if (r === Math.floor(rows / 2) && c === 0) div.classList.add('start');
        if (r === Math.floor(rows / 2) && c === cols - 1) div.classList.add('goal');
        if (r === agent.r && c === agent.c) div.classList.add('agent');
        gridEl.appendChild(div);
      }
    }
    const batPct = clamp(battery, 0, config.battery.max);
    const color = batPct > 50 ? '#22c55e' : batPct > 20 ? '#f59e0b' : '#ef4444';
    hud.innerHTML = `Battery <span class="battery"><div style="width:${batPct}%; background:${color}"></div></span> <span style="margin-left:6px">${batPct.toFixed(0)}%</span>`;
  }

  function neighbors(agent, walls) {
    const rows = config.grid.rows, cols = config.grid.cols;
    const dirs = [[-1, 0, 'UP'], [1, 0, 'DOWN'], [0, -1, 'LEFT'], [0, 1, 'RIGHT']];
    const out = [];
    for (const [dr, dc, name] of dirs) {
      const rr = agent.r + dr, cc = agent.c + dc;
      const key = `${rr},${cc}`;
      out.push({ dir: name, blocked: walls.has(key) || rr < 0 || rr >= rows || cc < 0 || cc >= cols });
    }
    return out;
  }

  function driftToNudge(drift) {
    const bias = 0.5 + (config._bias || 0.25) * Math.sign(drift);
    const gust = Math.random() < (config._gust_rate || 0.14) ? 2 : 1;
    return (Math.random() < bias ? 1 : -1) * gust;
  }

  // Build local map as 0/1 matrix (0=passable, 1=wall)
  function buildLocalMap(r, c, walls, grid_rows, grid_cols, rowRange = 2, colRange = 3) {
    const matrix = [];

    for (let dr = -rowRange; dr <= rowRange; dr++) {
      const row = [];
      for (let dc = -colRange; dc <= colRange; dc++) {
        const absRow = r + dr;
        const absCol = c + dc;

        // Out of bounds or wall = 1, otherwise = 0
        if (absRow < 0 || absRow >= grid_rows || absCol < 0 || absCol >= grid_cols) {
          row.push(1);
        } else if (walls.has(`${absRow},${absCol}`)) {
          row.push(1);
        } else {
          row.push(0);
        }
      }
      matrix.push(row);
    }

    return matrix;
  }

  // Removed API and LLM Prompts

  // Removed callAPI

  let isRunning = false;

  async function run() {
    if (isRunning) return;
    isRunning = true;

    // Disable controls
    startBtn.disabled = true;
    startBtn.innerText = 'now use your keyboard to move the drone';
    dlBtn.style.display = 'none';

    try {
      applyUI(); // read sliders
      runLog = { session: 'BatteryLayersV6', mode, trials: [], ui: getControls() };
      const blocks = defaults.blocks;

      for (let b = 0; b < blocks; b++) {
        const factors = {
          volatility: b % 2 === 0 ? 'low' : 'high',
          urgency: b % 2 === 0 ? 'off' : 'on',
          sensor_noise: b % 2 === 0 ? 'low' : 'high',
          map_density: b % 2 === 0 ? 'sparse' : 'dense'
        };
        const u = getControls();
        const trialsPerBlock = u.trials_per_block;
        for (let t = 0; t < trialsPerBlock; t++) {
          status.innerText = `Block ${b + 1}/${blocks} • Trial ${t + 1}/${trialsPerBlock} (${mode})`;
          await runOneTrial({ b, t, factors });
        }
      }
      status.innerText = 'All trials complete. You can download the data.';
    } catch (e) {
      console.error(e);
      status.innerText = 'Error during run: ' + e.message;
    } finally {
      isRunning = false;
      startBtn.disabled = false;
      startBtn.innerText = 'Restart';
      dlBtn.style.display = 'inline-block';
    }
  }

  function currentLayerDrift(drifts, r) {
    const gen = drifts[r];
    const { value } = gen.next();
    return value;
  }

  let beliefSliderAbortController = null;

  function showSurvey(reason) {
    return new Promise(resolve => {
      let chosenContext = null;
      document.querySelectorAll('.chip').forEach(x => x.classList.remove('active'));
      surveyModal.classList.add('active'); // Changed to use classList

      const reasonEl = document.getElementById('surveyReason');
      if (reasonEl) {
        const messages = {
          goal: 'Trial Complete! Goal Reached 🏁',
          battery: 'Battery Depleted 🔋',
          timeout_wallclock: 'Time Limit Reached ⏱️',
          timeout_idle: 'Idle Timeout — No Input Detected ⏱️'
        };
        reasonEl.innerText = messages[reason] || 'Trial Ended';
        reasonEl.style.color = reason === 'goal' ? 'var(--success)' : 'var(--danger)';
      }

      if (okSurvey) {
        okSurvey.disabled = true; // Disable until slider is moved
        if (window.StageHint) {
          StageHint.bindButton(okSurvey, 'Please adjust the belief slider first.');
        }
      }

      const beliefSliderRef = document.getElementById('beliefSlider');

      if (beliefSliderRef) {
        beliefSliderRef.value = "50";
        if (beliefVal) beliefVal.innerText = "50";

        if (beliefSliderAbortController) {
          beliefSliderAbortController.abort();
        }
        beliefSliderAbortController = new AbortController();

        beliefSliderRef.addEventListener('input', () => {
          if (beliefVal) beliefVal.innerText = beliefSliderRef.value;
          if (okSurvey) okSurvey.disabled = false;
        }, { signal: beliefSliderAbortController.signal });
      }

      const cleanup = () => {
        clearSurveyTimers();
        surveyModal.classList.remove('active'); // Changed to use classList
      };

      // ── Survey timeout: auto-submit with current value ──
      const sWallclockSec = 10;
      let sWallclockTimer = null;
      let timeRemaining = sWallclockSec;

      const timerDisplay = document.getElementById('surveyTimerDisplay');
      if (timerDisplay) {
        timerDisplay.style.visibility = 'visible';
        timerDisplay.innerText = `Time remaining: ${timeRemaining}s`;
      }

      function clearSurveyTimers() {
        if (sWallclockTimer) { clearInterval(sWallclockTimer); sWallclockTimer = null; }
      }

      function autoSubmitSurvey() {
        clearSurveyTimers();
        const val = Number(beliefSliderRef ? beliefSliderRef.value : 50);
        const result = { context_belief: val, auto_submitted: (timeRemaining <= 0) };
        cleanup(); resolve(result);
      }

      sWallclockTimer = setInterval(() => {
        timeRemaining--;
        if (timerDisplay) timerDisplay.innerText = `Time remaining: ${timeRemaining}s`;
        if (timeRemaining <= 0) {
          autoSubmitSurvey();
        }
      }, 1000);

      okSurvey.onclick = () => {
        const val = Number(beliefSliderRef ? beliefSliderRef.value : 50);
        const result = { context_belief: val };
        cleanup(); resolve(result);
      };
    });
  }

  async function runOneTrial({ b, t, factors }) {
    const walls = newMap(factors.map_density);
    const drifts = driftGenerators(factors.volatility);

    // Constants for this trial
    const goalRow = Math.floor(config.grid.rows / 2);
    const goalCol = config.grid.cols - 1;

    let agent = { r: goalRow, c: 0 };
    let step = 0;
    let time_left = config.trial.max_steps;
    let battery = config.battery.max;
    let currentBelief = 50; // default start
    let collisions = 0;
    let prevStep = { action: null, row: null, col: null };

    // History design:
    // - Navigation: no history (each step is independent for speed)
    // - Survey: fullHistory (complete record for evaluation)
    let fullHistory = [];

    const trialLog = { block: b, trial: t, factors, steps: [], mode };

    render(agent, walls, battery);

    const getPromptState = () => ({
      step,
      r: agent.r,
      c: agent.c,
      battery: Math.round(battery),
      time_left,
      collisions,
      urgency: factors.urgency,
      walls: walls,  // Pass the full walls set for local map generation
      prev_action: prevStep.action,
      prev_row: prevStep.row,
      prev_col: prevStep.col,
      grid_rows: config.grid.rows,
      grid_cols: config.grid.cols,
      goal_row: goalRow,
      goal_col: goalCol,
      current_belief: currentBelief
    });

    let awaiting = true;
    const trialAbortController = new AbortController();

    // ── Timeout configuration ──
    const timeoutCfg = defaults.timeout || {};
    const wallclockSec = timeoutCfg.trial_wallclock_sec || 90;
    const idleSec = timeoutCfg.idle_sec || 20;

    let wallclockTimer = null;
    let idleTimer = null;
    let countdownInterval = null;
    let trialStartTime = Date.now();
    let dsbIdleLeft = idleSec;

    // Countdown display on start button
    function updateCountdown() {
      const elapsed = (Date.now() - trialStartTime) / 1000;
      const remaining = Math.max(0, Math.ceil(wallclockSec - elapsed));
      startBtn.innerText = `⏱ ${remaining}s  |  idle ${dsbIdleLeft}s — use keyboard to move the drone`;
    }

    // Start wall-clock timer
    wallclockTimer = setTimeout(() => {
      if (awaiting) finish('timeout_wallclock');
    }, wallclockSec * 1000);

    // Start countdown interval (updates every second)
    updateCountdown();
    countdownInterval = setInterval(updateCountdown, 1000);

    function resetIdleTimer() {
      if (idleTimer) clearTimeout(idleTimer);
      dsbIdleLeft = idleSec;
      idleTimer = setTimeout(() => {
        if (awaiting) finish('timeout_idle');
      }, idleSec * 1000);
    }
    resetIdleTimer(); // Start initial idle timer
    // Idle countdown display
    let dsbIdleCountdown = setInterval(() => { dsbIdleLeft = Math.max(0, dsbIdleLeft - 1); }, 1000);

    // Helper to clear all timers
    function clearAllTimers() {
      if (wallclockTimer) { clearTimeout(wallclockTimer); wallclockTimer = null; }
      if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
      if (countdownInterval) { clearInterval(countdownInterval); countdownInterval = null; }
      if (dsbIdleCountdown) { clearInterval(dsbIdleCountdown); dsbIdleCountdown = null; }
    }

    const onKey = (e) => {
      if (mode !== 'human' || !awaiting) return;
      if (e.repeat) return; // Ignore if the key is being held down
      if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) return;
      e.preventDefault();
      resetIdleTimer(); // Reset idle timer on each valid key press
      const mapKey = { 'ArrowUp': 'UP', 'ArrowDown': 'DOWN', 'ArrowLeft': 'LEFT', 'ArrowRight': 'RIGHT' };
      stepOnce(mapKey[e.key], '');
    };
    document.addEventListener('keydown', onKey, { signal: trialAbortController.signal });

    // Global idle resets (mousemove/scroll)
    let lastGlobalIdleReset = Date.now();
    const globalIdleReset = () => {
      if (Date.now() - lastGlobalIdleReset > 1000) {
        lastGlobalIdleReset = Date.now();
        resetIdleTimer();
      }
    };
    document.addEventListener('mousemove', globalIdleReset, { signal: trialAbortController.signal });
    document.addEventListener('scroll', globalIdleReset, { signal: trialAbortController.signal });

    function applyDrain(collided) {
      let d = config.battery.drain_per_step + (collided ? config.battery.drain_collision_penalty : 0);
      if (factors.urgency === 'on') d += (config.battery.urgency_extra || 0);
      battery = Math.max(0, battery - d);

      if (collided) {
        gridEl.classList.remove('shake');
        void gridEl.offsetWidth; // trigger reflow
        gridEl.classList.add('shake');
        setTimeout(() => gridEl.classList.remove('shake'), 300);
      }
    }

    function atGoal() {
      return (agent.r === goalRow && agent.c === goalCol);
    }

    function endCondition() {
      if (atGoal()) return 'goal';
      if (battery <= 0) return 'battery';
      if (time_left <= 0) return 'timeout';
      return null;
    }

    function stepOnce(actionLabel, rationale) {
      if (!awaiting) return; // Prevent moves after end

      let dr = 0, dc = 0;
      if (actionLabel === 'UP') dr = -1;
      if (actionLabel === 'DOWN') dr = 1;
      if (actionLabel === 'LEFT') dc = -1;
      if (actionLabel === 'RIGHT') dc = 1;

      // 1. Proposed Move
      const nextR = clamp(agent.r + dr, 0, config.grid.rows - 1);
      const nextC = clamp(agent.c + dc, 0, config.grid.cols - 1);

      // 2. Check Wall Collision (Voluntary Move)
      let collision = 0;
      let hitWall = false;

      if (walls.has(`${nextR},${nextC}`)) {
        hitWall = true;
        collision = 1;
        // Agent stays put
      } else {
        agent.r = nextR;
        agent.c = nextC;
      }

      // 3. Apply Wind Drift
      let actualDrift = 0;
      let drift = 0;
      let nudge = 0;

      // Apply drift only based on probability
      if (Math.random() < (config.drift_probability !== undefined ? config.drift_probability : 1.0)) {
        drift = currentLayerDrift(drifts, agent.r);
        nudge = driftToNudge(drift);

        // Iterative drift check to prevent tunneling
        const driftSteps = Math.abs(nudge);
        const driftDir = Math.sign(nudge);

        for (let i = 1; i <= driftSteps; i++) {
          const checkC = agent.c + (driftDir * i);

          // Check bounds
          if (checkC < 0 || checkC >= config.grid.cols) {
            collision = 1; // Hit boundary
            break;
          }

          // Check wall
          if (walls.has(`${agent.r},${checkC}`)) {
            collision = 1; // Hit wall
            break;
          }

          // If safe, we can move here
          actualDrift = driftDir * i;
        }
      }

      agent.c += actualDrift;

      step += 1;
      time_left -= 1;

      collisions += collision;
      applyDrain(collision > 0); // Pass boolean or count? applyDrain takes boolean-ish
      render(agent, walls, battery);

      trialLog.steps.push({
        step, action: actionLabel, rationale: rationale || null,
        row: agent.r, col: agent.c, drift_row: drift, nudge, battery: Math.round(battery),
        pos: `${agent.r},${agent.c}`, collision, belief: currentBelief
      });

      const why = endCondition();
      if (why) {
        finish(why);
      } else {
        prevStep = { action: actionLabel, row: agent.r, col: agent.c };
      }
    }

    function finish(why) {
      if (!awaiting) return; // Already finished
      awaiting = false;
      clearAllTimers(); // Clean up all timeout/idle timers
      trialAbortController.abort(); // Destroy the keydown listener

      if (why === 'timeout_wallclock') {
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

      if (why === 'timeout_idle') {
        let strikes = parseInt(sessionStorage.getItem('dsb_idle_strikes') || '0', 10);
        strikes++;
        sessionStorage.setItem('dsb_idle_strikes', strikes.toString());

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

      trialLog.end_reason = why;
      trialLog.elapsed_sec = parseFloat(((Date.now() - trialStartTime) / 1000).toFixed(1));
      runLog.trials.push(trialLog);

      // Auto-save to localStorage
      try {
        localStorage.setItem('dsb_backup_log', JSON.stringify(runLog));
      } catch (e) { console.warn('Backup failed', e); }

      proceed(why);
    }

    async function proceed(why) {
      showSurvey(why).then(async res => {
        trialLog.context_belief = res.context_belief;

        const u = getControls();
        const totalTrials = defaults.blocks * u.trials_per_block;
        const isLastTrial = (runLog.trials.length >= totalTrials);

        if (isLastTrial) {
          // Show Task Complete Modal
          const completeModal = document.getElementById('completeModal');
          const saveStatus = document.getElementById('saveStatus');
          if (completeModal) completeModal.classList.add('active');

          // Save data to localStorage (unified submission happens on main menu)
          try {
            // Add subject ID to the run log
            runLog.subjectId = window.SUBJECT_ID || 'unknown';
            localStorage.setItem('experiment_data_dsb', JSON.stringify(runLog));
            // Mark DSB as complete for main menu indicator
            localStorage.setItem('task_complete_dsb', 'true');

            if (saveStatus) saveStatus.innerText = "Data saved successfully!";
            if (saveStatus) saveStatus.style.color = "var(--success)";
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
            if (saveStatus) saveStatus.innerText = "Error saving data. Backup saved locally.";
            if (saveStatus) saveStatus.style.color = "var(--danger)";
            const subAll = document.getElementById('downloadBtn');
            if (subAll) subAll.style.display = 'inline-block';
          }
        }

        resolveTrial();
      });
    }

    let resolveTrial;
    const done = new Promise(res => { resolveTrial = res; });

    // Let the keydown logic call `finish(why)` which then calls `proceed(why)`,
    // and `proceed` calls `resolveTrial()` resolving `done`.
    await done;
  }

  // Initial grid CSS sizing
  (function initGrid() {
    const r = defaults.grid.rows, c = defaults.grid.cols, cell = defaults.grid.cell;
    gridEl.style.gridTemplateRows = `repeat(${r}, ${cell}px)`;
    gridEl.style.gridTemplateColumns = `repeat(${c}, ${cell}px)`;
  })();

  // Practice Trial Logic
  async function runPracticeTrial() {
    if (isRunning) return;
    isRunning = true;
    startBtn.disabled = true;
    startBtn.innerText = 'Practice...';

    const goalRow = Math.floor(config.grid.rows / 2);
    const goalCol = config.grid.cols - 1;

    // Hardcode a practice map so there's an inescapable wall 
    // right near the start position, preventing an empty sparse generation.
    const pWalls = new Set();
    pWalls.add(`${goalRow},2`);
    pWalls.add(`${goalRow - 1},2`);
    pWalls.add(`${goalRow + 1},2`);
    pWalls.add(`${goalRow - 1},1`);
    pWalls.add(`${goalRow + 1},1`);

    let pAgent = { r: goalRow, c: 0 };
    let pBattery = 100; // Force full battery visual
    render(pAgent, pWalls, pBattery);

    let stage = 1; // 1 = crash, 2 = reach goal

    if (window.dsbTour && window.dsbTour.currentStep === 0) {
      window.dsbTour.nextStep();
    }

    let pAwaiting = true;
    const pAbort = new AbortController();

    // ── Practice timeouts ──
    const pTimeoutCfg = defaults.timeout || {};
    const pWallclockSec = pTimeoutCfg.practice_wallclock_sec || 10;
    const pIdleSec = pTimeoutCfg.practice_idle_sec || 10;
    let pWallclockTimer = null;
    let pIdleTimer = null;

    function clearPracticeTimers() {
      if (pWallclockTimer) { clearTimeout(pWallclockTimer); pWallclockTimer = null; }
      if (pIdleTimer) { clearTimeout(pIdleTimer); pIdleTimer = null; }
    }

    function resetPracticeIdleTimer() {
      if (pIdleTimer) clearTimeout(pIdleTimer);
      pIdleTimer = setTimeout(() => {
        if (pAwaiting) autoAdvancePractice();
      }, pIdleSec * 1000);
    }

    function startPracticeWallclock() {
      if (pWallclockTimer) clearTimeout(pWallclockTimer);
      pWallclockTimer = setTimeout(() => {
        if (pAwaiting) autoAdvancePractice();
      }, pWallclockSec * 1000);
      resetPracticeIdleTimer();
    }

    function autoAdvancePractice() {
      if (!pAwaiting) return;
      clearPracticeTimers();

      if (stage === 1) {
        // Auto-simulate a collision to advance to stage 2
        stage = 2;
        pAwaiting = false;
        gridEl.classList.remove('shake');
        void gridEl.offsetWidth;
        gridEl.classList.add('shake');
        setTimeout(() => gridEl.classList.remove('shake'), 300);

        setTimeout(() => {
          pAgent.r = goalRow;
          pAgent.c = goalCol - 2;
          render(pAgent, pWalls, pBattery);
          pAwaiting = true;
          if (window.dsbTour) window.dsbTour.nextStep();
          startPracticeWallclock(); // Restart timers for stage 2
        }, 800);
      } else if (stage === 2) {
        // Auto-teleport to goal to advance to stage 3
        pAgent.r = goalRow;
        pAgent.c = goalCol;
        render(pAgent, pWalls, pBattery);
        // Trigger goal logic
        stepP('RIGHT'); // This will detect goal and trigger stage 3
        // If stepP didn't fire (already at goal), force it:
        if (stage === 2) {
          stage = 3;
          pAwaiting = false;
          if (pAbort) pAbort.abort();

          const tt = document.querySelector('.tour-tooltip');
          if (tt) tt.classList.remove('show');
          const overlay = document.querySelector('.tour-overlay');
          if (overlay) overlay.style.opacity = '0';
          if (window.dsbTour) window.dsbTour.nextStep();

          const sModal = document.getElementById('surveyModal');
          if (sModal) {
            sModal.setAttribute('style', 'z-index: 100000 !important; visibility: visible !important; opacity: 1 !important; display: flex !important;');
            sModal.classList.add('active');
          }
          setTimeout(() => {
            showSurvey('goal').then(res => {
              if (window.dsbTour) window.dsbTour.finish();
              isRunning = false;
              startBtn.disabled = true;
              if (sModal) sModal.removeAttribute('style');
              resolveP();
            });
          }, 300);
        }
      }
    }

    // Start practice timers for stage 1
    startPracticeWallclock();

    function onKeyP(e) {
      if (!pAwaiting) return;
      if (e.repeat) return;
      if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) return;
      e.preventDefault();
      resetPracticeIdleTimer(); // Reset idle on key press
      const mapKey = { 'ArrowUp': 'UP', 'ArrowDown': 'DOWN', 'ArrowLeft': 'LEFT', 'ArrowRight': 'RIGHT' };
      stepP(mapKey[e.key]);
    }

    document.addEventListener('keydown', onKeyP, { signal: pAbort.signal });

    function stepP(action) {
      if (!pAwaiting) return;
      let dr = 0, dc = 0;
      if (action === 'UP') dr = -1;
      if (action === 'DOWN') dr = 1;
      if (action === 'LEFT') dc = -1;
      if (action === 'RIGHT') dc = 1;

      const nextR = clamp(pAgent.r + dr, 0, config.grid.rows - 1);
      const nextC = clamp(pAgent.c + dc, 0, config.grid.cols - 1);

      let collision = false;
      if (pWalls.has(`${nextR},${nextC}`)) {
        collision = true;
      } else {
        pAgent.r = nextR;
        pAgent.c = nextC;
      }

      let drain = config.battery.drain_per_step + (collision ? config.battery.drain_collision_penalty : 0);
      pBattery = Math.max(30, pBattery - drain); // Don't let them actually die in practice

      if (collision) {
        gridEl.classList.remove('shake');
        void gridEl.offsetWidth;
        gridEl.classList.add('shake');
        setTimeout(() => gridEl.classList.remove('shake'), 300);
      }

      render(pAgent, pWalls, pBattery);

      if (stage === 1 && collision) {
        stage = 2; // Move to stage 2
        pAwaiting = false; // pause briefly
        clearPracticeTimers();
        setTimeout(() => {
          // Teleport near goal
          pAgent.r = goalRow;
          pAgent.c = goalCol - 2;
          render(pAgent, pWalls, pBattery);
          pAwaiting = true;
          if (window.dsbTour) window.dsbTour.nextStep(); // Advance "Fly to Goal"
          startPracticeWallclock(); // Restart timers for stage 2
        }, 800);
      } else if (stage === 2 && pAgent.r === goalRow && pAgent.c === goalCol) {
        stage = 3; // Ensure this block only runs exactly once
        pAwaiting = false;
        clearPracticeTimers();
        if (pAbort) pAbort.abort();

        // Brute force hide tooltips in case of error
        const tt = document.querySelector('.tour-tooltip');
        if (tt) tt.classList.remove('show');
        const overlay = document.querySelector('.tour-overlay');
        if (overlay) overlay.style.opacity = '0';

        if (window.dsbTour) window.dsbTour.nextStep(); // Advance to survey

        // Force modal via injected CSS to absolutely guarantee it overrides everything
        const sModal = document.getElementById('surveyModal');
        if (sModal) {
          sModal.setAttribute('style', 'z-index: 100000 !important; visibility: visible !important; opacity: 1 !important; display: flex !important;');
          sModal.classList.add('active');
        }

        setTimeout(() => {
          showSurvey('goal').then(res => {
            if (window.dsbTour) window.dsbTour.finish(); // End tour
            // Completely resolve practice logic
            isRunning = false;
            startBtn.disabled = true; // wait for real run to enable it or just immediately trigger run

            // Clean up styles
            if (sModal) sModal.removeAttribute('style');
            resolveP();
          });
        }, 300); // Slight delay to ensure DOM updates and tour transitions finish
      }
    }

    let resolveP;
    const doneP = new Promise(res => { resolveP = res; });
    await doneP;
  }

  // Buttons
  let practiceDone = false;
  startBtn.onclick = async () => {
    if (!practiceDone && window.dsbTour) {
      practiceDone = true;
      // Start practice mode natively. Make sure real run() is not active.
      if (isRunning) return;
      await runPracticeTrial();
      // After practice, auto-start the real trial!
      setTimeout(() => {
        run();
      }, 500);
    } else {
      run();
    }
  };
  dlBtn.onclick = () => {
    const blob = new Blob([JSON.stringify(runLog, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dsb_layers_v6_run_${Date.now()}.json`;
    document.body.appendChild(a); a.click(); a.remove();
  };

  // Auto-submission removed — data is saved to localStorage and submitted from main menu.
  // const WEBHOOK_URL = (removed — unified submission from index.html);

  // Removed API Configuration UI

})();
