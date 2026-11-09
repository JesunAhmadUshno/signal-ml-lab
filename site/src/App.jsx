import { useMemo, useState, useEffect, useId } from 'react';
import learn from '../../level-1-learn/results_learn.json';
import build from '../../level-2-build/results_build.json';
import deep from '../../level-3-deep/results_deep.json';
import rigor from '../../level-3-deep/results_rigor.json';
import manifest from '../../level-2-build/artifacts/model/1/manifest.json';
import model from './generated/model.json';
import { signals, labels, params } from './generated/testset.json';
import { features, forward, quantizeModel, score, rocAuc, rng, makeSignal, probe, FEATURE_NAMES, LENGTH, RECIPE, PROBE_PER_START } from './signal.js';

const REPO = 'https://github.com/JesunAhmadUshno/signal-ml-lab';
const blob = (p) => `${REPO}/blob/main/${p}`;
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const fmt = (n) => Math.round(n).toLocaleString('en-US');
const f3 = (x) => x.toFixed(3);

// Test count is read from the committed test files at build time.
const testFiles = import.meta.glob('../../tests/test_*.py', { query: '?raw', import: 'default', eager: true });
const TEST_COUNT = Object.values(testFiles).reduce((n, src) => n + (src.match(/^\s+def test_/gm) || []).length, 0);

const testFeats = signals.map((s) => features(s).f);
const qModel = quantizeModel(model);
const N_TEST = labels.length;
const N_DEFECT = labels.filter((y) => y === 1).length, N_CLEAN = N_TEST - N_DEFECT;
const N_TRAIN = manifest.training_config.train_signals;
const edge = deep.edge_study;
const lat = deep.measured_timing;
const tr = build.training;

const SECTIONS = [['top', 'Overview', '◎'], ['lab', 'Signal lab', '∿'], ['replay', 'Test set', '▦'], ['sweep', 'Where it breaks', '◐'], ['rigor', 'Baselines', '≡'], ['edge', 'Edge study', '⚡'], ['limits', 'Limits', '!'], ['reproduce', 'Reproduce', '❯_']];

function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) { els.forEach((e) => e.classList.add('in')); return; }
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.08 });
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, []);
}

function MenuBar() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => { const t = setInterval(() => setNow(new Date()), 30000); return () => clearInterval(t); }, []);
  return (
    <div className="menubar" role="banner">
      <span className="apple" aria-hidden="true">◆</span>
      <a className="app" href="#top">Signal ML Lab</a>
      <a className="m" href={blob('PROPOSAL.md')}>Proposal</a>
      <a className="m" href={blob('level-3-deep/JUDGMENT.md')}>Judgment</a>
      <a className="m" href={blob('REPORT.md')}>Report</a>
      <span className="spacer" />
      <span className="status"><i /> {TEST_COUNT} tests in CI</span>
      <a href={REPO}>GitHub ↗</a>
      <span className="clock">{now.toLocaleString('en-US', { weekday: 'short', hour: 'numeric', minute: '2-digit' })}</span>
    </div>
  );
}

function Dock() {
  const [active, setActive] = useState('top');
  useEffect(() => {
    const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && setActive(e.target.id)), { rootMargin: '-45% 0px -50% 0px' });
    SECTIONS.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);
  return (
    <nav className="dock" aria-label="Sections">
      {SECTIONS.map(([id, label, icon]) => (
        <a key={id} href={`#${id}`} className={active === id ? 'on' : ''} aria-current={active === id ? 'true' : undefined}>
          <span className="ico" aria-hidden="true">{icon}</span><span className="lbl">{label}</span>
        </a>
      ))}
    </nav>
  );
}

function Range({ label, value, min, max, step, onChange, show }) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label} <span>{show}</span></label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </div>
  );
}

function Seg({ label, value, options, onChange }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map(([k, l]) => <button key={String(k)} aria-pressed={value === k} onClick={() => onChange(k)}>{l}</button>)}
    </div>
  );
}

// Small trace chart: signal polyline, optional base and defect band.
function Trace({ sig, base, defect, label, H = 200, W = 560 }) {
  const pad = { l: 30, r: 8, t: 10, b: 20 };
  const lo = -2.6, hi = 2.6;
  const x = (i) => pad.l + (i / (LENGTH - 1)) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (H - pad.t - pad.b);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      {[-2, -1, 0, 1, 2].map((t) => <g key={t}><line className="axis" x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} /><text x={pad.l - 6} y={y(t) + 4} textAnchor="end">{t}</text></g>)}
      {defect && <rect x={x(defect.start) - 3} width={x(Math.min(LENGTH - 1, defect.start + defect.width - 1)) - x(defect.start) + 6} y={pad.t} height={H - pad.t - pad.b} fill="var(--bad)" opacity="0.1" />}
      {base && <polyline fill="none" stroke="var(--muted)" strokeDasharray="4 4" strokeWidth="1.2" points={base.map((v, i) => `${x(i)},${y(v)}`).join(' ')} />}
      <polyline fill="none" stroke="var(--primary)" strokeWidth="2" points={sig.map((v, i) => `${x(i)},${y(v)}`).join(' ')} />
      <text x={pad.l} y={H - 4}>sample 0</text><text x={W - pad.r} y={H - 4} textAnchor="end">sample {LENGTH - 1}</text>
    </svg>
  );
}

function Coeffs({ name, v, scale, hot }) {
  const W = 260, H = 96, mid = H / 2, bw = (W - 8) / v.length;
  return (
    <figure className="coef">
      <figcaption><span className="mono">{name}</span> <span className="hint">{v.length} coefficients</span></figcaption>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Haar ${name} coefficients`}>
        <line className="axis" x1={0} x2={W} y1={mid} y2={mid} />
        {v.map((c, i) => {
          const h = Math.max(-1, Math.min(1, c / scale)) * (mid - 4);
          return <rect key={i} x={4 + i * bw + 0.5} width={Math.max(1, bw - 1)} y={h > 0 ? mid - h : mid} height={Math.abs(h)} rx="1" fill={hot(i) ? 'var(--bad)' : 'var(--primary)'} opacity={hot(i) ? 0.95 : 0.6} />;
        })}
      </svg>
    </figure>
  );
}

export default function App() {
  useReveal();
  return (
    <div className="wrap">
      <MenuBar />
      <div className="glow" aria-hidden="true" />
      <Hero />
      <SignalLab />
      <Replay />
      <Sweep />
      <Rigor />
      <Training />
      <Pipeline />
      <Edge />
      <Artifact />
      <Limits />
      <Reproduce />
      <Dock />
      <footer>
        Synthetic data only, generated by the repo's own recipe. Code: Apache-2.0. Every number on this page is read at build time from the committed <code>results_*.json</code> files, the artifact manifest and the exported test set; the signal lab, test-set scoring, int8 toggle and difficulty probe run in your browser on the committed weights.
      </footer>
    </div>
  );
}

function Hero() {
  const ci = rigor.headline_committed_model.wilson95;
  const correct = Math.round(tr.test_accuracy * build.test_counts.total);
  return (
    <header className="card hero window" id="top" data-title="signal-ml-lab — README">
      <div className="head">
        <p className="eyebrow">Haar wavelets · hand-written NumPy network · live inference</p>
        <span className="pill warn">● Synthetic data only</span>
      </div>
      <h1>A {(edge.float_artifact.size_on_disk.total_bytes / 1000).toFixed(1)} KB wavelet network <em>sorts {correct} of {build.test_counts.total}</em> synthetic signals, and it runs in your browser.</h1>
      <p className="sub">
        A clean public rebuild of a 2023 fabric-inspection method: twelve Haar wavelet features feed a two-layer network written by hand in NumPy.
        Every signal here is synthetic, a stand-in for inspection signals and not the paper's data. The page loads the committed model weights and scores the exact {N_TEST}-signal test set again in your browser.
        Learn reached {learn.accuracy.toFixed(3)}, but only on {learn.test_counts.total} easy signals, so it does not lead here.
      </p>
      <div className="actions">
        <a className="btn primary" href="#lab">Open the signal lab</a>
        <a className="btn ghost" href="#sweep">See where it breaks</a>
        <span className="hint">Reproduce with <code>python level-2-build/run_build.py</code></span>
      </div>
      <div className="tiles">
        <div className="tile"><div className="num">{tr.test_accuracy.toFixed(3)}</div><div className="cap">Build test accuracy, {correct}/{build.test_counts.total} (95% CI {ci[0].toFixed(3)} to {ci[1].toFixed(3)})</div></div>
        <div className="tile"><div className="num">{fmt(edge.float_artifact.size_on_disk.total_bytes)}</div><div className="cap">bytes for the whole versioned artifact, manifest included</div></div>
        <div className="tile"><div className="num">{lat.latency_inprocess.mean_ms}<small> ms</small></div><div className="cap">mean in-process prediction, {lat.latency_inprocess.calls} calls, Muse VM CPU (varies by run)</div></div>
        <div className="tile"><div className="num">{TEST_COUNT}</div><div className="cap">unit tests in <code>tests/</code>, run in CI on Python 3.11 and 3.12</div></div>
      </div>
    </header>
  );
}

function SignalLab() {
  const [noise, setNoise] = useState(0.05);
  const [on, setOn] = useState(true);
  const [amp, setAmp] = useState(1.1);
  const [width, setWidth] = useState(5);
  const [start, setStart] = useState(26);
  const [seed, setSeed] = useState(7);
  const st = Math.min(start, LENGTH - width);
  const defect = on ? { start: st, width, amp } : null;
  const sig = useMemo(() => makeSignal(rng(seed), { noise, defect }), [seed, noise, on, amp, width, st]); // eslint-disable-line react-hooks/exhaustive-deps
  const base = useMemo(() => makeSignal(rng(seed), { noise: 0 }), [seed]);
  const { f, coeffs } = features(sig);
  const out = forward(model, f);
  const inBump = (i, step) => defect && i * step + step > defect.start && i * step < defect.start + defect.width;
  const scale = (v) => Math.max(0.4, ...v.map(Math.abs));
  const aligned = on && width === 4 && st % 4 === 0;
  return (
    <section id="lab" className="card window reveal" data-title="signal_lab — live">
      <div className="head">
        <div>
          <p className="eyebrow">Live, in your browser</p>
          <h2>Draw a signal, add a bump, and watch the <em>wavelet features</em> decide.</h2>
          <p className="sub">
            The generator follows the repo's <a href={blob('level-1-learn/signal_ml.py')}>make_dataset</a> recipe: two sines, Gaussian noise, and an optional local bump.
            Two Haar levels split it into approximation and detail coefficients, the twelve features are standardised with the committed mean and spread, and the committed network scores it.
            Your browser's random numbers are not NumPy's, so these are new samples from the same recipe, not the recorded ones.
          </p>
        </div>
        <span className={`pill ${out.pred ? 'bad' : 'good'}`}>{out.pred ? '● Defect' : '● Clean'} · p(defect) {out.p[1].toFixed(3)}</span>
      </div>
      <div className="controls">
        <Range label="Noise σ" value={noise} min={0} max={0.3} step={0.01} onChange={setNoise} show={noise.toFixed(2)} />
        <Range label="Bump amplitude" value={amp} min={-1.6} max={1.6} step={0.05} onChange={setAmp} show={amp.toFixed(2)} />
        <Range label="Bump width" value={width} min={1} max={8} step={1} onChange={setWidth} show={`${width} samples`} />
        <Range label="Bump start" value={st} min={0} max={LENGTH - width} step={1} onChange={setStart} show={`sample ${st}`} />
      </div>
      <div className="actions" style={{ marginTop: 0, marginBottom: 16 }}>
        <Seg label="Defect on or off" value={on} options={[[true, 'Defect on'], [false, 'Clean signal']]} onChange={setOn} />
        <button className="btn ghost" onClick={() => setSeed((s) => s + 1)}>New noise sample</button>
        <button className="btn ghost" onClick={() => { setOn(true); setWidth(4); setStart(24); setAmp(1.1); setNoise(0.05); }}>Try the blind spot</button>
        <span className="hint">{aligned ? 'Aligned: this width-4 bump fills one level-2 Haar block, so both detail levels cancel it.' : `Training used noise ${RECIPE.noise}, amplitude ${RECIPE.amp[0]} to ${RECIPE.amp[1]}, width ${RECIPE.width[0]} to ${RECIPE.width[1] - 1}, start ${RECIPE.start[0]} to ${RECIPE.start[1] - 1}.`}</span>
      </div>
      <div className="grid2">
        <div>
          <h3>Signal (dashed: the same sines without noise or bump)</h3>
          <Trace sig={sig} base={base} defect={defect} label="Generated signal trace" />
          <h3 style={{ marginTop: 12 }}>Network output</h3>
          <div className="bars" style={{ marginTop: 10 }}>
            {[['Clean', out.p[0], 'var(--good)'], ['Defect', out.p[1], 'var(--bad)']].map(([k, v, c]) => (
              <div className="bar-row" key={k}>
                <div className="lab"><span>{k}</span><b>{pct(v, 1)}</b></div>
                <div className="track"><div className="fill" style={{ width: `${v * 100}%`, background: c }} /></div>
              </div>
            ))}
          </div>
        </div>
        <div className="coefgrid">
          <Coeffs name="a1" v={coeffs.a1} scale={scale(coeffs.a1)} hot={(i) => inBump(i, 2)} />
          <Coeffs name="d1" v={coeffs.d1} scale={scale(coeffs.d1)} hot={(i) => inBump(i, 2)} />
          <Coeffs name="a2" v={coeffs.a2} scale={scale(coeffs.a2)} hot={(i) => inBump(i, 4)} />
          <Coeffs name="d2" v={coeffs.d2} scale={scale(coeffs.d2)} hot={(i) => inBump(i, 4)} />
        </div>
      </div>
      <h3 style={{ marginTop: 18 }}>The twelve features, standardised (z = how far from the training mean)</h3>
      <div className="featgrid" style={{ marginTop: 10 }}>
        {FEATURE_NAMES.map((n, j) => {
          const z = out.x[j], w = Math.min(1, Math.abs(z) / 6) * 50;
          return (
            <div className="feat" key={n}>
              <span className="mono">{n}</span>
              <div className="ztrack"><div style={{ left: z < 0 ? `${50 - w}%` : '50%', width: `${w}%`, background: Math.abs(z) > 3 ? 'var(--bad)' : 'var(--primary)' }} /></div>
              <b>{z.toFixed(1)}</b>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function Replay() {
  const [thr, setThr] = useState(0.5);
  const [pick, setPick] = useState(0);
  const s = useMemo(() => score(model, testFeats, labels, thr), [thr]);
  const at05 = useMemo(() => score(model, testFeats, labels, 0.5), []);
  const auc = useMemo(() => rocAuc(at05.p, labels), [at05]);
  const parity = at05.correct === Math.round(tr.test_accuracy * N_TEST) && at05.cm.tp === tr.per_class_test['1'].correct;
  const cases = useMemo(() => labels.map((y, i) => ({ i, y, p: s.p[i], wrong: s.pred[i] !== y })).filter((c) => c.wrong || Math.abs(c.p - thr) < 0.1).sort((a, b) => b.wrong - a.wrong || Math.abs(a.p - thr) - Math.abs(b.p - thr)), [s, thr]);
  const sel = cases[Math.min(pick, Math.max(0, cases.length - 1))];
  const roc = useMemo(() => {
    const pts = [];
    for (let k = 0; k <= 100; k++) {
      const t = k / 100; let tp = 0, fp = 0;
      labels.forEach((y, i) => { if (at05.p[i] >= t) (y ? tp++ : fp++); });
      pts.push([fp / N_CLEAN, tp / N_DEFECT]);
    }
    return pts;
  }, [at05]);
  const nWrong = cases.filter((c) => c.wrong).length;
  const W = 260, H = 220, pad = 28;
  const rx = (v) => pad + v * (W - pad - 8), ry = (v) => H - pad - v * (H - pad - 8);
  const cur = [s.cm.fp / N_CLEAN, s.cm.tp / N_DEFECT];
  return (
    <section id="replay" className="card window reveal" data-title={`testset_replay.json — ${N_TEST} signals (recorded inputs, live inference)`}>
      <div className="head">
        <div>
          <p className="eyebrow">Recorded inputs · live inference</p>
          <h2>The exact seed-42 test set, <em>scored again</em> in your browser.</h2>
          <p className="sub">
            These {N_TEST} signals are the ones Build and Deep tested on, exported once from Python. Your browser recomputes every Haar feature and every prediction with the committed weights, then checks the count against <a href={blob('level-2-build/results_build.json')}>results_build.json</a>.
            Move the decision threshold to trade missed defects against false alarms.
          </p>
        </div>
        <span className={`pill ${parity ? 'good' : 'bad'}`}>{parity ? '✓' : '✗'} Parity at 0.5: {at05.correct}/{N_TEST} vs recorded {Math.round(tr.test_accuracy * N_TEST)}</span>
      </div>
      <div className="controls" style={{ gridTemplateColumns: 'minmax(0, 1fr)' }}>
        <Range label="Decision threshold on p(defect)" value={thr} min={0.01} max={0.99} step={0.01} onChange={(v) => { setThr(v); setPick(0); }} show={thr.toFixed(2)} />
      </div>
      <div className="grid3">
        <div>
          <h3>Confusion matrix · accuracy {pct(s.correct / N_TEST, 1)}</h3>
          <table className="cm" style={{ marginTop: 8 }}>
            <thead><tr><th scope="col">Truth \ predicted</th><th scope="col">Clean</th><th scope="col">Defect</th></tr></thead>
            <tbody>
              <tr><th scope="row">Clean</th><td className="ok">{s.cm.tn}</td><td className={s.cm.fp ? 'no' : ''}>{s.cm.fp}</td></tr>
              <tr><th scope="row">Defect</th><td className={s.cm.fn ? 'no' : ''}>{s.cm.fn}</td><td className="ok">{s.cm.tp}</td></tr>
            </tbody>
          </table>
          <p className="hint">At 0.5 the misses are all one-sided: every clean signal is right, {N_DEFECT - at05.cm.tp} defects slip through.</p>
        </div>
        <div>
          <h3>ROC curve · AUC {auc.toFixed(4)}</h3>
          <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="ROC curve of the committed model on the test set">
            {[0, 0.5, 1].map((t) => <g key={t}><line className="axis" x1={rx(0)} x2={rx(1)} y1={ry(t)} y2={ry(t)} /><text x={pad - 6} y={ry(t) + 4} textAnchor="end">{t}</text><text x={rx(t)} y={H - 10} textAnchor="middle">{t}</text></g>)}
            <line x1={rx(0)} y1={ry(0)} x2={rx(1)} y2={ry(1)} stroke="var(--grid)" strokeDasharray="4 4" />
            <polyline fill="none" stroke="var(--primary)" strokeWidth="2" points={roc.map(([a, b]) => `${rx(a)},${ry(b)}`).join(' ')} />
            <circle cx={rx(cur[0])} cy={ry(cur[1])} r="5" fill="var(--bad)" stroke="var(--surface-strong)" strokeWidth="2" />
            <text x={rx(1)} y={ry(0) - 6} textAnchor="end">false-positive rate →</text>
          </svg>
        </div>
        <div>
          <h3>Misses and close calls ({nWrong} wrong, {cases.length - nWrong} within 0.1)</h3>
          {sel ? (
            <>
              <Trace sig={signals[sel.i]} defect={params[sel.i]} label={`Test signal ${sel.i}`} H={150} W={300} />
              <div className="tip" aria-live="polite">
                #{sel.i}: truth <b>{sel.y ? 'defect' : 'clean'}</b>, p(defect) {sel.p.toFixed(3)} {sel.wrong ? <span className="pill bad">wrong</span> : <span className="pill warn">close</span>}
                {params[sel.i] && <> · amp {params[sel.i].amp.toFixed(2)}, width {params[sel.i].width}, start {params[sel.i].start}{params[sel.i].width === 4 && params[sel.i].start % 4 === 0 ? ' (aligned block)' : ''}</>}
              </div>
              <div className="actions" style={{ marginTop: 8 }}>
                <button className="btn ghost" onClick={() => setPick((p) => (p - 1 + cases.length) % cases.length)} aria-label="Previous case">← Prev</button>
                <span className="hint">{Math.min(pick, cases.length - 1) + 1} / {cases.length}</span>
                <button className="btn ghost" onClick={() => setPick((p) => (p + 1) % cases.length)} aria-label="Next case">Next →</button>
              </div>
            </>
          ) : <p className="hint">No misses or close calls at this threshold.</p>}
        </div>
      </div>
    </section>
  );
}

function Heat({ cells, fpr, title, amps, noises }) {
  const W = 620, cw = 48, ch = 34, l = 78, t = 22, H = t + noises.length * ch + 26;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
      {amps.map((a, i) => <text key={a} x={l + i * cw + cw / 2} y={14} textAnchor="middle">{a}</text>)}
      <text x={l + amps.length * cw + 30} y={14} textAnchor="middle">false alarms</text>
      {noises.map((nz, r) => (
        <g key={nz}>
          <text x={l - 8} y={t + r * ch + ch / 2 + 4} textAnchor="end">σ {nz}</text>
          {amps.map((a, i) => {
            const v = cells.find((c) => c.noise === nz && c.amp === a)?.detection_rate ?? 0;
            return (
              <g key={a}>
                <rect x={l + i * cw + 1} y={t + r * ch + 1} width={cw - 2} height={ch - 2} rx="5" fill="var(--primary)" opacity={0.08 + 0.85 * v} />
                <text x={l + i * cw + cw / 2} y={t + r * ch + ch / 2 + 4} textAnchor="middle" style={{ fill: v > 0.55 ? '#fff' : 'var(--text)' }}>{Math.round(v * 100)}</text>
              </g>
            );
          })}
          <rect x={l + amps.length * cw + 6} y={t + r * ch + 1} width={48} height={ch - 2} rx="5" fill="var(--bad)" opacity={0.08 + 0.8 * fpr[r]} />
          <text x={l + amps.length * cw + 30} y={t + r * ch + ch / 2 + 4} textAnchor="middle" style={{ fill: fpr[r] > 0.55 ? '#fff' : 'var(--text)' }}>{Math.round(fpr[r] * 100)}</text>
        </g>
      ))}
      <text x={l} y={H - 4}>bump amplitude →  (cells: % of defects detected)</text>
    </svg>
  );
}

function Sweep() {
  const sw = rigor.difficulty_sweep;
  const [view, setView] = useState('recorded');
  const [round, setRound] = useState(0);
  const live = useMemo(() => {
    const cells = [], fpr = [];
    sw.noises.forEach((nz, r) => {
      sw.amps.forEach((a) => cells.push({ noise: nz, amp: a, detection_rate: probe(model, { amp: a, noise: nz, n: 200, seed: 1000 * round + Math.round(a * 100) * 7 + Math.round(nz * 100) }).detection }));
      fpr[r] = probe(model, { amp: 1, noise: nz, n: 200, seed: 99 + round }).fpr;
    });
    const starts = sw.start_width4_at_nominal.map(({ start }) => probe(model, { amp: RECIPE.amp, noise: RECIPE.noise, n: PROBE_PER_START, seed: 500 + start + 1000 * round, width: 4, start }).detection);
    return { cells, fpr, starts };
  }, [round, sw]);
  const faint = Math.max(...sw.amp_noise.filter((c) => c.noise === sw.noises[0] && c.detection_rate < 0.5).map((c) => c.amp));
  const recFpr = sw.noises.map((nz) => sw.false_positive_rate_by_noise[String(nz)]);
  const W = 960, H = 200, pad = { l: 40, r: 10, t: 10, b: 24 };
  const xs = (s) => pad.l + (s / 60) * (W - pad.l - pad.r), ys = (v) => pad.t + (1 - v) * (H - pad.t - pad.b);
  return (
    <section id="sweep" className="card window reveal" data-title="difficulty_sweep — recorded (Python) vs in-browser probe">
      <div className="head">
        <div>
          <p className="eyebrow">Where the method breaks</p>
          <h2>Two failure modes the {tr.test_accuracy.toFixed(3)} hides: <em>noise</em> and <em>alignment</em>.</h2>
          <p className="sub">
            v1.0 only tested the training conditions. The rigor pass (<a href={blob('level-3-deep/rigor.py')}>rigor.py</a>) swept bump size and noise with fresh synthetic sets and the committed model.
            At σ {sw.noises[0]}, bumps of amplitude {faint} or less are mostly missed. Worse, the model has never seen noise above σ {RECIPE.noise}: at σ {sw.noises[1]} it already raises false alarms on {pct(recFpr[1], 0)} of clean signals, and from σ {sw.noises[2]} it calls everything a defect, so those detection rates mean nothing.
            Switch to the live probe to regenerate the grid in your browser with a different random generator.
          </p>
        </div>
        <Seg label="Sweep source" value={view} options={[['recorded', 'Recorded (Python)'], ['live', 'Live probe (browser)']]} onChange={setView} />
      </div>
      <Heat title={`Detection rate by amplitude and noise, ${view}`} cells={view === 'live' ? live.cells : sw.amp_noise} fpr={view === 'live' ? live.fpr : recFpr} amps={sw.amps} noises={sw.noises} />
      <div className="actions" style={{ marginTop: 6 }}>
        <button className="btn ghost" onClick={() => { setView('live'); setRound((r) => r + 1); }}>Re-run live probe with new seeds</button>
        <span className="hint">{view === 'live' ? `Live: ${sw.signals_per_cell} defect and ${sw.signals_per_cell} clean signals per noise level, generated and scored just now.` : `Recorded: ${sw.signals_per_cell} signals per cell, seeded, in results_rigor.json.`}</span>
      </div>
      <h3 style={{ marginTop: 22 }}>Detection by bump start, width 4, training amplitude and noise</h3>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Detection rate by start position for width-4 bumps, recorded line and live dots" style={{ marginTop: 8 }}>
        {[0, 0.5, 1].map((t) => <g key={t}><line className="axis" x1={pad.l} x2={W - pad.r} y1={ys(t)} y2={ys(t)} /><text x={pad.l - 6} y={ys(t) + 4} textAnchor="end">{pct(t, 0)}</text></g>)}
        {[0, 8, 16, 24, 32, 40, 48, 56].map((s) => <text key={s} x={xs(s)} y={H - 6} textAnchor="middle">{s}</text>)}
        <polyline fill="none" stroke="var(--primary)" strokeWidth="2" points={sw.start_width4_at_nominal.map((d) => `${xs(d.start)},${ys(d.detection_rate)}`).join(' ')} />
        {live.starts.map((v, i) => <circle key={i} cx={xs(sw.start_width4_at_nominal[i].start)} cy={ys(v)} r="3" fill="var(--warn)" />)}
      </svg>
      <div className="legend"><span><i style={{ background: 'var(--primary)' }} />recorded (Python, {sw.signals_per_cell} per start)</span><span><i style={{ background: 'var(--warn)' }} />live probe (browser, {PROBE_PER_START} per start)</span></div>
      <p className="hint" style={{ marginTop: 8 }}>
        Every dip sits on a multiple of 4. A width-4 bump that starts there fills one level-2 Haar block exactly, so the level-1 and level-2 detail coefficients cancel it and only the approximation features move.
        {' '}{rigor.error_analysis.missed_width4_start_multiple_of_4} of the {rigor.error_analysis.missed_defect} test-set misses are this case, although only {rigor.error_analysis.test_defects_width4_start_multiple_of_4} of {rigor.error_analysis.test_defects_total} test defects are aligned.
      </p>
    </section>
  );
}

const BASE_LABELS = {
  majority_class: 'Always say "clean"',
  single_feature_threshold: 'One threshold on one feature',
  logistic_regression_12_wavelet_features: 'Logistic regression, 12 wavelet features',
  smallnn_raw_64_samples: 'Same network, raw 64 samples',
  smallnn_fft_12_band_energies: 'Same network, 12 FFT band energies',
  smallnn_12_wavelet_features_refit: 'Same network, 12 wavelet features (v1.0)',
};

function Rigor() {
  const b = rigor.baselines;
  const rows = Object.keys(BASE_LABELS).map((k) => ({ k, ...b[k] }));
  const lo = 0.4, sx = (v) => `${((v - lo) / (1 - lo)) * 100}%`;
  const seeds = rigor.seed_variation;
  const thrB = b.single_feature_threshold;
  return (
    <section id="rigor" className="card window reveal" data-title="rigor.py — results_rigor.json">
      <p className="eyebrow">Baselines, budget and seeds</p>
      <h2>The wavelet network is good, but <em>not special</em> on this task.</h2>
      <p className="sub">
        Same train set ({fmt(N_TRAIN)} signals), same {N_TEST}-signal test set, each with a 95% Wilson interval. One threshold on feature {thrB.feature_index} (level-2 detail energy) already gets {f3(thrB.accuracy)}.
        A raw-signal network and logistic regression land inside the v1.0 interval, and FFT features score {f3(b.smallnn_fft_12_band_energies.accuracy)}, so this synthetic task does not show that wavelet features help.
      </p>
      <div className="bars" style={{ marginTop: 18 }}>
        {rows.map((r) => (
          <div className="bar-row" key={r.k}>
            <div className="lab"><span>{BASE_LABELS[r.k]}</span><b>{f3(r.accuracy)} <span className="hint">({r.correct}/{r.total}, CI {f3(r.wilson95[0])} to {f3(r.wilson95[1])})</span></b></div>
            <div className="track">
              <div className="fill" style={{ width: sx(Math.max(lo, r.accuracy)), background: r.k.endsWith('refit') ? 'var(--primary)' : 'color-mix(in srgb, var(--muted) 55%, transparent)' }} />
              <div className="ci" style={{ left: sx(Math.max(lo, r.wilson95[0])), width: `calc(${sx(Math.max(lo, r.wilson95[1]))} - ${sx(Math.max(lo, r.wilson95[0]))})` }} />
            </div>
          </div>
        ))}
      </div>
      <p className="hint" style={{ marginTop: 8 }}>Bar scale {lo} to 1.0. The dark band is the 95% Wilson interval.</p>
      <div className="grid2" style={{ marginTop: 20 }}>
        <div>
          <h3>Training longer closes the gap</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ marginTop: 8 }}>
              <thead><tr><th>Learning rate</th><th>Epochs</th><th>Test accuracy</th><th>95% CI</th></tr></thead>
              <tbody>{rigor.training_budget.map((r) => <tr key={`${r.lr}-${r.epochs}`}><td>{r.lr}</td><td>{fmt(r.epochs)}</td><td className={r.accuracy === 1 ? 'ok' : ''}>{f3(r.accuracy)} ({r.correct}/{N_TEST})</td><td>{f3(r.wilson95[0])} to {f3(r.wilson95[1])}</td></tr>)}</tbody>
            </table>
          </div>
          <p className="hint">The v1.0 headline (first row) measures an under-trained network, not the method's ceiling.</p>
        </div>
        <div>
          <h3>Seed variation</h3>
          <SeedStrip label="Training seeds 0 to 9, same data" vals={seeds.training_seeds_0_to_9_accuracy} />
          <SeedStrip label="Five fresh train and test sets" vals={seeds.data_seeds.map((d) => d.accuracy)} />
          <p className="hint">Training-seed mean {f3(seeds.training_seed_mean)} (sd {seeds.training_seed_std}); fresh-data mean {f3(seeds.data_seed_mean)}. The seed-42 test set is a little easier than average.</p>
        </div>
      </div>
    </section>
  );
}

function SeedStrip({ label, vals }) {
  const W = 420, H = 54, lo = 0.96, hi = 1;
  const x = (v) => 12 + ((v - lo) / (hi - lo)) * (W - 24);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${label}: test accuracies`} style={{ marginTop: 8 }}>
      <text x={12} y={12} style={{ fill: 'var(--text)' }}>{label}</text>
      <line className="axis" x1={12} x2={W - 12} y1={30} y2={30} />
      {vals.map((v, i) => <circle key={i} cx={x(v)} cy={30 + ((i % 3) - 1) * 5} r="5" fill="var(--primary)" opacity="0.7" />)}
      <line x1={x(tr.test_accuracy)} x2={x(tr.test_accuracy)} y1={20} y2={40} stroke="var(--bad)" strokeWidth="2" />
      {[lo, 0.98, hi].map((t) => <text key={t} x={x(t)} y={H - 2} textAnchor="middle">{t.toFixed(2)}</text>)}
    </svg>
  );
}

function Training() {
  const series = [[`Learn (${fmt(learn.train_counts.total)} train signals)`, learn.loss_curve, 'var(--warn)'], [`Build (${fmt(N_TRAIN)} train signals)`, tr.loss_curve, 'var(--primary)']];
  const lastEpoch = Math.max(...series.flatMap(([, pts]) => pts.map((p) => p.epoch)));
  const W = 960, H = 240, pad = { l: 44, r: 16, t: 12, b: 26 };
  const x = (e) => pad.l + ((e - 1) / (lastEpoch - 1)) * (W - pad.l - pad.r), y = (v) => pad.t + (1 - v / 0.6) * (H - pad.t - pad.b);
  return (
    <section id="train" className="card window reveal" data-title="loss_curve.svg — recorded">
      <p className="eyebrow">Recorded training runs</p>
      <h2>Hand-written gradient descent, <em>{tr.epochs} epochs</em>, still falling at the end.</h2>
      <p className="sub">
        Full-batch training loss as recorded in <a href={blob('level-1-learn/results_learn.json')}>results_learn.json</a> and <a href={blob('level-2-build/results_build.json')}>results_build.json</a> (learning rate {tr.lr}, logged every {tr.loss_curve[2].epoch - tr.loss_curve[1].epoch} epochs).
        Learn goes from {learn.loss_curve[0].loss} to {learn.loss_curve.at(-1).loss}, Build from {tr.loss_curve[0].loss} to {tr.loss_curve.at(-1).loss}. Neither curve has flattened, which matches the training-budget result above.
      </p>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Recorded training loss by epoch for Learn and Build" style={{ marginTop: 12 }}>
        {[0, 0.2, 0.4, 0.6].map((t) => <g key={t}><line className="axis" x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} /><text x={pad.l - 6} y={y(t) + 4} textAnchor="end">{t}</text></g>)}
        {series.map(([k, pts, c]) => (
          <g key={k}>
            <polyline fill="none" stroke={c} strokeWidth="2.5" points={pts.map((p) => `${x(p.epoch)},${y(p.loss)}`).join(' ')} />
            {pts.map((p) => <g key={p.epoch}><circle cx={x(p.epoch)} cy={y(p.loss)} r="4.5" fill={c} stroke="var(--surface-strong)" strokeWidth="2" /><text x={x(p.epoch)} y={y(p.loss) - 9} textAnchor="middle">{p.loss}</text></g>)}
          </g>
        ))}
        <text x={pad.l} y={H - 6}>epoch 1</text><text x={W - pad.r} y={H - 6} textAnchor="end">epoch {tr.epochs}</text>
      </svg>
      <div className="legend">{series.map(([k, , c]) => <span key={k}><i style={{ background: c }} />{k}</span>)}</div>
    </section>
  );
}

function Pipeline() {
  const m = build.measured_timing, p = build.pipeline;
  const max = m.naive_two_pass_seconds;
  return (
    <section id="bench" className="card window reveal" data-title="pipeline_bench — recorded">
      <p className="eyebrow">Recorded benchmark · Muse VM, varies by run</p>
      <h2>Parallel extraction plus a cache: <em>{m.speedup_two_pass}x</em> faster over two passes.</h2>
      <p className="sub">Two full passes over the {fmt(build.train_counts.total)} training signals. The naive loader extracts features on both passes; the optimised pipeline extracts once across worker processes and serves the second pass from cache. Both produce identical features ({p.features_equal_naive_vs_optimized ? 'checked' : 'not checked'}).</p>
      <div className="bars" style={{ marginTop: 16 }}>
        {[['Naive loop, two passes', m.naive_two_pass_seconds, 'var(--muted)'], ['Parallel map + cache, two passes', m.optimized_two_pass_seconds, 'var(--primary)']].map(([k, v, c]) => (
          <div className="bar-row" key={k}><div className="lab"><span>{k}</span><b>{v} s</b></div><div className="track"><div className="fill" style={{ width: `${(v / max) * 100}%`, background: c }} /></div></div>
        ))}
      </div>
      <div className="tiles">
        <div className="tile"><div className="num">{p.workers}</div><div className="cap">worker processes, chunks of {p.chunk_size} signals</div></div>
        <div className="tile"><div className="num">{p.batch_count}</div><div className="cap">prefetched batches of {p.batch_size} (seed {p.batch_seed}), {fmt(p.batch_total_rows)} rows</div></div>
        <div className="tile"><div className="num">{p.cache_hits}/{p.cache_misses}</div><div className="cap">cache hits / misses across loads</div></div>
        <div className="tile"><div className="num">{fmt(p.extraction_count_first_pipeline)}</div><div className="cap">feature extractions, done once</div></div>
      </div>
    </section>
  );
}

function Edge() {
  const [mode, setMode] = useState('float');
  const fl = useMemo(() => score(model, testFeats, labels), []);
  const q8 = useMemo(() => score(qModel, testFeats, labels), []);
  const cur = mode === 'float' ? fl : q8;
  const flips = fl.pred.filter((v, i) => v !== q8.pred[i]).length;
  const maxDp = Math.max(...fl.p.map((v, i) => Math.abs(v - q8.p[i])));
  const rec = mode === 'float' ? edge.float_artifact.correct : edge.int8_quantized_numpy_study.correct;
  const files = Object.entries(edge.float_artifact.size_on_disk.per_file_bytes).sort((a, b) => b[1] - a[1]);
  const fmax = files[0][1];
  const qz = edge.int8_quantized_numpy_study;
  const prof = deep.profiler_note;
  return (
    <section id="edge" className="card window reveal" data-title="edge_study.py — float vs int8">
      <div className="head">
        <div>
          <p className="eyebrow">Edge study on CPU · NumPy quantization, not TFLite</p>
          <h2>Squeeze the weights to int8 and <em>nothing flips</em>.</h2>
          <p className="sub">
            The toggle applies <a href={blob('level-3-deep/edge.py')}>edge.py</a>'s per-array int8 quantization (scale = max|w| / 127) to the real weights in your browser and re-scores all {N_TEST} test signals.
            Sizes and latencies are recorded on the Muse VM CPU and imported from <a href={blob('level-3-deep/results_deep.json')}>results_deep.json</a>.
          </p>
        </div>
        <Seg label="Weight precision" value={mode} options={[['float', 'float64'], ['int8', 'int8']]} onChange={setMode} />
      </div>
      <div className="tiles">
        <div className="tile"><div className="num">{cur.correct}/{N_TEST}</div><div className="cap">correct, recomputed live ({rec} recorded) {cur.correct === rec ? '✓' : '✗'}</div></div>
        <div className="tile"><div className="num">{flips}</div><div className="cap">predictions that change between float and int8</div></div>
        <div className="tile"><div className="num">{maxDp.toFixed(4)}</div><div className="cap">largest change in p(defect) on any test signal</div></div>
        <div className="tile"><div className="num">{qz.size_ratio_quantized_over_float}</div><div className="cap">int8 / float array bytes ({fmt(qz.quantized_size_bytes)} vs {fmt(qz.float_array_bytes_comparable)})</div></div>
      </div>
      <div className="grid2" style={{ marginTop: 20 }}>
        <div>
          <h3>Float artifact on disk, {fmt(edge.float_artifact.size_on_disk.total_bytes)} bytes</h3>
          <div className="bars" style={{ marginTop: 10 }}>
            {files.map(([k, v]) => <div className="bar-row" key={k}><div className="lab"><span className="mono">{k}</span><b>{fmt(v)} B</b></div><div className="track"><div className="fill" style={{ width: `${(v / fmax) * 100}%`, background: k.endsWith('.json') ? 'var(--warn)' : 'var(--primary)' }} /></div></div>)}
          </div>
          <p className="hint">The int8 size counts .npy files with headers plus one scale per array, so small arrays shrink less than 8x.</p>
        </div>
        <div>
          <h3>Latency per prediction (recorded, single run)</h3>
          <div className="bars" style={{ marginTop: 10 }}>
            {[['In-process, mean', lat.latency_inprocess.mean_ms, lat.latency_inprocess.calls], ['In-process, p95', lat.latency_inprocess.p95_ms, lat.latency_inprocess.calls], ['HTTP POST /predict, mean', lat.latency_http.mean_ms, lat.latency_http.calls], ['HTTP POST /predict, p95', lat.latency_http.p95_ms, lat.latency_http.calls]].map(([k, v, n]) => (
              <div className="bar-row" key={k}><div className="lab"><span>{k} <span className="hint">({n} calls)</span></span><b>{v} ms</b></div><div className="track"><div className="fill" style={{ width: `${(v / lat.latency_http.p95_ms) * 100}%`, background: k.startsWith('HTTP') ? 'var(--warn)' : 'var(--primary)' }} /></div></div>
            ))}
          </div>
          <p className="hint">The HTTP layer, not the model, dominates: {lat.http_overhead_ratio_mean.toFixed(1)}x the in-process mean on one local machine.</p>
        </div>
      </div>
      <h3 style={{ marginTop: 20 }}>Profiler top steps (Python cProfile, not TF Profiler): {prof.slowest_step.function} takes {pct(prof.slowest_step_tottime_share, 1)} of own time</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ marginTop: 8 }}>
          <thead><tr><th>Function</th><th>File</th><th>Calls</th><th>Own time (s)</th><th>Cumulative (s)</th></tr></thead>
          <tbody>{prof.top_steps.slice(0, 5).map((r) => <tr key={r.function + r.line}><td className="mono">{r.function}</td><td className="mono">{r.file}</td><td>{fmt(r.calls)}</td><td>{r.tottime_s}</td><td>{r.cumtime_s}</td></tr>)}</tbody>
        </table>
      </div>
    </section>
  );
}

function Artifact() {
  const sizes = edge.float_artifact.size_on_disk.per_file_bytes;
  return (
    <section id="artifact" className="card window reveal" data-title="artifact/manifest.json">
      <p className="eyebrow">Versioned model artifact</p>
      <h2>A model that <em>refuses to load</em> if anything was touched.</h2>
      <div className="grid2" style={{ marginTop: 16 }}>
        <div className="manifest" role="group" aria-label="Manifest fields">
          <dl>
            <dt>format</dt><dd>{manifest.format}</dd>
            <dt>version</dt><dd>{manifest.version}</dd>
            <dt>input</dt><dd>{manifest.input_signature.feature_dim} features, {manifest.input_signature.dtype}</dd>
            <dt>output</dt><dd>{manifest.output_signature.classes} classes: {manifest.output_signature.labels.join(', ')}</dd>
            <dt>training</dt><dd>{fmt(manifest.training_config.train_signals)} signals, hidden {manifest.training_config.hidden}, {manifest.training_config.epochs} epochs, lr {manifest.training_config.lr}, seed {manifest.training_config.seed}</dd>
            <dt>sha256</dt><dd className="hash">{manifest.weights_sha256}</dd>
            <dt>files</dt><dd>{manifest.weight_files.map((w) => `${w} (${fmt(sizes[w])} B)`).join(' · ')}</dd>
          </dl>
        </div>
        <div>
          <h3>What <span className="mono">load_artifact</span> rejects</h3>
          <ul className="checks">
            <li>a missing manifest, an unknown format string, or a version mismatch</li>
            <li>an input signature that is not {manifest.input_signature.feature_dim} features or an output that is not {manifest.output_signature.classes} classes</li>
            <li>a changed weight-file list or a missing weight file</li>
            <li>any byte change in the weights: the SHA-256 over the six files in order must match</li>
            <li>array shapes that disagree with the signatures</li>
          </ul>
          <p className="hint">The format is local to this repo, not SavedModel, and loadable by nothing else. The weights this page runs on carry the same hash ({model.weights_sha256 === manifest.weights_sha256 ? 'matches' : 'does not match'}).</p>
        </div>
      </div>
    </section>
  );
}

function Limits() {
  const sw = rigor.difficulty_sweep;
  const items = [
    ['Synthetic data only', 'No public fabric dataset and not the 2023 paper data. The published method is rebuilt, not validated against its published results.', 'bad'],
    ['No TensorFlow', `TensorFlow was not importable (${build.environment.note.split(';')[0]}). TF Serving, TFLite and TF Profiler are named substitutions: a stdlib HTTP API, a NumPy int8 study and cProfile.`, 'warn'],
    ['CPU only, one machine', 'Nothing ran on a GPU or real edge hardware. Latencies are single runs on the Muse VM CPU and vary between runs.', 'warn'],
    ['An easy, narrow task', `One threshold gets ${f3(rigor.baselines.single_feature_threshold.accuracy)} and longer training reaches ${f3(rigor.training_budget.at(-1).accuracy)}. The model breaks under unseen noise (σ ${sw.noises[1]}: ${pct(sw.false_positive_rate_by_noise[String(sw.noises[1])], 0)} false alarms) and on block-aligned bumps.`, 'warn'],
    ['Train and test share a generator', 'There is no distribution-shift test beyond the sweep above, and no related-work comparison with published numbers yet.', 'warn'],
    ['Browser parity', 'The test-set replay is exact (checked at build time against the Python predictions). The live lab and probe use a different random generator, so they show the recipe, not the recorded samples.', 'good'],
  ];
  return (
    <section id="limits" className="card window reveal" data-title="JUDGMENT.md — limits">
      <p className="eyebrow">Limitations and threats to validity</p>
      <h2>What this repo <em>does not</em> show.</h2>
      <div className="grid2" style={{ marginTop: 16 }}>
        {items.map(([t, d, k]) => (
          <div className="tile" key={t}>
            <div className="head" style={{ marginBottom: 6 }}><h3>{t}</h3><span className={`pill ${k}`}>{k === 'good' ? 'Checked' : k === 'bad' ? 'Main threat' : 'Caveat'}</span></div>
            <p className="hint" style={{ margin: 0, fontSize: 13 }}>{d}</p>
          </div>
        ))}
      </div>
      <p className="hint" style={{ marginTop: 14 }}>Each remaining gap and the data or compute it needs is in <a href={blob('REPORT.md')}>REPORT.md</a>.</p>
    </section>
  );
}

function Reproduce() {
  const docs = [
    ['PROPOSAL.md', 'Proposal', 'Question, scope, draft plan (not pre-registered)'],
    ['level-1-learn/README.md', 'Level 1 Learn', 'Haar features, NumPy network'],
    ['level-2-build/README.md', 'Level 2 Build', 'Pipeline, training, artifact'],
    ['level-3-deep/README.md', 'Level 3 Deep', 'API and edge study'],
    ['level-3-deep/JUDGMENT.md', 'Judgment', 'When TensorFlow is the right tool'],
    ['REPORT.md', 'Report', 'Results, misses, threats, gaps'],
  ];
  return (
    <section id="reproduce" className="card window reveal" data-title="Terminal — reproduce">
      <p className="eyebrow">Reproduce it</p>
      <h2>NumPy and the standard library, nothing else.</h2>
      <ol className="steps" style={{ marginTop: 16 }}>
        <li><span className="n">1</span><div><b>Install</b><div className="hint"><code>git clone {REPO}.git && cd signal-ml-lab && pip install numpy</code></div></div></li>
        <li><span className="n">2</span><div><b>Run the three levels</b><div className="hint"><code>python level-1-learn/run_learn.py && python level-2-build/run_build.py && python level-3-deep/run_deep.py</code></div></div></li>
        <li><span className="n">3</span><div><b>Rigor pass and dashboard data</b><div className="hint"><code>python level-3-deep/rigor.py && python level-3-deep/export_site_data.py</code></div></div></li>
        <li><span className="n">4</span><div><b>Run the {TEST_COUNT} tests</b><div className="hint"><code>python -m unittest discover -s tests</code></div></div></li>
      </ol>
      <div className="docs" style={{ marginTop: 20 }}>
        {docs.map(([p, t, d]) => <a key={p} href={blob(p)}><b>{t}</b><small>{d}</small></a>)}
      </div>
    </section>
  );
}

