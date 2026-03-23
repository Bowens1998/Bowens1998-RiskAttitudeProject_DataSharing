// Analysis logic ported from dsb_complete_analysis.ipynb

// --- Math Helpers ---
function mean(arr) {
    if (!arr.length) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr) {
    if (arr.length < 2) return 0;
    const m = mean(arr);
    const variance = arr.reduce((a, b) => a + Math.pow(b - m, 2), 0) / (arr.length - 1);
    return Math.sqrt(variance);
}

function zscore(arr) {
    const m = mean(arr);
    const s = std(arr);
    if (s === 0) return arr.map(() => 0);
    return arr.map(x => (x - m) / s);
}

// --- Bayesian Filter ---
const D = [-2, -1, 0, 1, 2];

function priorUniform() {
    return D.map(() => 1.0 / D.length);
}

function transitionMat(alpha = 0.1) {
    const n = D.length;
    const T = [];
    for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < n; j++) {
            if (i === j) {
                row.push(1 - alpha);
            } else {
                row.push(alpha / (n - 1));
            }
        }
        T.push(row);
    }
    return T;
}

function likelihoodDiscrete(nudge, eps = 0.2) {
    // P(o | d): prefer sign(d) == nudge, otherwise eps
    let like = D.map(d => {
        const signD = Math.sign(d);
        const signNudge = Math.sign(nudge);
        let match = false;
        if (nudge !== 0) {
            match = (signD === signNudge);
        } else {
            match = (d === 0);
        }
        return match ? (1 - eps) : eps;
    });

    // Smoothing
    like = like.map(x => 0.001 + 0.999 * x);

    // Normalize
    const sum = like.reduce((a, b) => a + b, 0);
    return like.map(x => x / sum);
}

function matVecMul(vec, mat) {
    const res = new Array(vec.length).fill(0);
    for (let j = 0; j < vec.length; j++) {
        for (let i = 0; i < vec.length; i++) {
            res[j] += vec[i] * mat[i][j];
        }
    }
    return res;
}

function runBayesFilter(steps) {
    const alpha = 0.15;
    const eps = 0.25;
    const T = transitionMat(alpha);

    // Beliefs per row: { rowNum: [prob_d-2, ..., prob_d2] }
    let beliefs = {};
    const rows = [...new Set(steps.map(s => s.row))];
    rows.forEach(r => beliefs[r] = priorUniform());

    const results = [];

    steps.forEach(s => {
        const r = s.row;
        let prior = beliefs[r] || priorUniform();

        // Predict: prior * T
        prior = matVecMul(prior, T);

        // Update
        let post;
        if (s.nudge !== undefined && s.nudge !== null) {
            const like = likelihoodDiscrete(s.nudge, eps);
            post = prior.map((p, i) => p * like[i]);
            const sum = post.reduce((a, b) => a + b, 0);
            post = post.map(p => p / sum);
        } else {
            post = prior;
        }

        beliefs[r] = post;

        // MAP estimate
        let maxP = -1;
        let maxIdx = -1;
        post.forEach((p, i) => {
            if (p > maxP) {
                maxP = p;
                maxIdx = i;
            }
        });
        const mapDrift = D[maxIdx];

        // Entropy
        const ent = -post.reduce((sum, p) => sum + (p * Math.log(p + 1e-12)), 0);

        results.push({
            step: s.step,
            row: r,
            nudge: s.nudge,
            drift_row_true: s.drift_row,
            map_drift: mapDrift,
            entropy: ent
        });
    });

    return results;
}

// --- Main Analysis Function ---
function performAnalysis(runLog) {
    if (!runLog || !runLog.trials || runLog.trials.length === 0) {
        return { error: "No trial data available." };
    }

    const trials = runLog.trials;
    const metrics = [];

    // 1. Calculate Per-Trial Metrics
    trials.forEach(tr => {
        const steps = tr.steps;
        const totalSteps = steps.length;
        const collisions = steps.filter(s => s.collision).length;
        const finalBattery = steps.length > 0 ? steps[steps.length - 1].battery : 0;

        // TTG: Time to Goal (steps)
        const TTG = totalSteps;

        // CR: Collision Rate
        const CR = collisions / Math.max(1, totalSteps);

        // DE: Drift Error (mean absolute drift)
        const drifts = steps.map(s => Math.abs(s.drift_row || 0));
        const DE = mean(drifts);

        // BM: Battery Margin (final battery)
        const BM = finalBattery;

        metrics.push({
            block: tr.block,
            trial: tr.trial,
            end_reason: tr.end_reason,
            human_label: tr.human_context_label || tr.llm_context_label,
            conf: tr.human_context_conf || tr.llm_context_conf || 0,
            TTG, CR, DE, BM
        });
    });

    // 2. Calculate Caution Index (CI)
    const zTTG = zscore(metrics.map(m => m.TTG));
    const zCR = zscore(metrics.map(m => m.CR));
    const zDE = zscore(metrics.map(m => m.DE));
    const zBM = zscore(metrics.map(m => m.BM));

    metrics.forEach((m, i) => {
        m.CI = zTTG[i] + (-zCR[i]) + (-zDE[i]) + (-zBM[i]);
    });

    // 3. Run Bayes Filter for the last trial (as a sample)
    const lastTrial = trials[trials.length - 1];
    const bayesResults = runBayesFilter(lastTrial.steps);

    return {
        metrics: metrics,
        bayes_sample: bayesResults,
        trial_count: trials.length
    };
}

// --- UI Rendering ---
function renderAnalysis(analysis) {
    if (analysis.error) return `<div style="color:red">${analysis.error}</div>`;

    let html = `<h3>Analysis Results (${analysis.trial_count} trials)</h3>`;

    // Metrics Table
    html += `<h4>Trial Metrics & Caution Index</h4>`;
    html += `<table border="1" style="border-collapse:collapse; width:100%; font-size:12px; text-align:center;">
    <tr style="background:#f1f5f9;">
      <th>Trial</th><th>End</th><th>Belief (0-100)</th><th>TTG</th><th>CR</th><th>DE</th><th>BM</th><th>CI</th>
    </tr>`;

    analysis.metrics.forEach(m => {
        const ciColor = m.CI > 0 ? '#dcfce7' : '#fee2e2';
        // Handle backward compatibility or new field
        const belief = (m.context_belief !== undefined) ? m.context_belief : (m.human_label || '-');

        html += `<tr>
      <td>${m.block}-${m.trial}</td>
      <td>${m.end_reason}</td>
      <td>${belief}</td>
      <td>${m.TTG}</td>
      <td>${m.CR.toFixed(2)}</td>
      <td>${m.DE.toFixed(2)}</td>
      <td>${m.BM}</td>
      <td style="background:${ciColor}">${m.CI.toFixed(2)}</td>
    </tr>`;
    });
    html += `</table>`;

    // Bayes Filter Chart (Simplified as table for now)
    html += `<h4>Bayesian Belief Filter (Last Trial Sample)</h4>`;
    html += `<div style="max-height:200px; overflow-y:auto;">
  <table border="1" style="border-collapse:collapse; width:100%; font-size:12px; text-align:center;">
    <tr style="background:#f1f5f9;">
      <th>Step</th><th>Row</th><th>Nudge</th><th>True Drift</th><th>MAP Est</th><th>Entropy</th>
    </tr>`;

    analysis.bayes_sample.forEach(b => {
        const match = b.map_drift === b.drift_row_true;
        const color = match ? '#dcfce7' : '#fee2e2';
        html += `<tr>
      <td>${b.step}</td>
      <td>${b.row}</td>
      <td>${b.nudge}</td>
      <td>${b.drift_row_true}</td>
      <td style="background:${color}">${b.map_drift}</td>
      <td>${b.entropy.toFixed(2)}</td>
    </tr>`;
    });
    html += `</table></div>`;

    return html;
}
