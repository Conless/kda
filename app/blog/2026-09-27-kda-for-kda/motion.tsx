'use client';

import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';
import {
  accuracyCute,
  accuracyFlashKda,
  accuracyTirx,
  speedupAllShapes,
  speedupFocused,
  tokensAllShapes,
  tokensFocused,
} from './chart-data';

type Series = readonly (readonly [number, number])[];

const InViewContext = createContext(false);

function prefersReducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function useInView<T extends Element>(threshold: number) {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        setInView(true);
        observer.disconnect();
      }
    }, { threshold });
    observer.observe(node);
    return () => observer.disconnect();
  }, [threshold]);

  return [ref, inView] as const;
}

/** Adds `is-in` once the block scrolls into view; CSS keys every entrance animation off it. */
export function Reveal({
  className = '',
  threshold = 0.25,
  label,
  children,
}: {
  className?: string;
  threshold?: number;
  label?: string;
  children: ReactNode;
}) {
  const [ref, inView] = useInView<HTMLDivElement>(threshold);
  return (
    <div
      ref={ref}
      className={`reveal${inView ? ' is-in' : ''}${className ? ` ${className}` : ''}`}
      role={label ? 'figure' : undefined}
      aria-label={label}
    >
      <InViewContext.Provider value={inView}>{children}</InViewContext.Provider>
    </div>
  );
}

/** Counts from `from` to `to` when the enclosing Reveal enters the viewport. Renders the final value for no-JS readers. */
export function CountUp({
  to,
  from = 0,
  decimals = 0,
  suffix = '',
  duration = 1400,
  delay = 0,
  linear = false,
}: {
  to: number;
  from?: number;
  decimals?: number;
  suffix?: string;
  duration?: number;
  delay?: number;
  linear?: boolean;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const play = useContext(InViewContext);
  const format = (value: number) =>
    `${value.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}${suffix}`;

  useEffect(() => {
    const node = ref.current;
    if (!node || prefersReducedMotion()) return;
    const render = (value: number) => {
      node.textContent = `${value.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}${suffix}`;
    };
    if (!play) {
      render(from);
      return;
    }
    let frame = 0;
    const timer = window.setTimeout(() => {
      const start = performance.now();
      const tick = (now: number) => {
        const t = Math.min(1, (now - start) / duration);
        const eased = linear ? t : 1 - (1 - t) ** 3;
        render(from + (to - from) * eased);
        if (t < 1) frame = requestAnimationFrame(tick);
      };
      frame = requestAnimationFrame(tick);
    }, delay);
    return () => {
      window.clearTimeout(timer);
      cancelAnimationFrame(frame);
    };
  }, [play, from, to, decimals, suffix, duration, delay, linear]);

  return <span ref={ref}>{format(to)}</span>;
}

function linePath(series: Series, x: (value: number) => number, y: (value: number) => number) {
  return series.map(([a, b], index) => `${index ? 'L' : 'M'}${x(a).toFixed(1)} ${y(b).toFixed(1)}`).join('');
}

function stepPath(series: Series, x: (value: number) => number, y: (value: number) => number) {
  return series
    .map(([a, b], index) => {
      if (!index) return `M${x(a).toFixed(1)} ${y(b).toFixed(1)}`;
      return `H${x(a).toFixed(1)}V${y(b).toFixed(1)}`;
    })
    .join('');
}

const workloads = [
  { label: 'H96 fixed', detail: '1 × 8192', cute: 2.76, tirx: 3.11 },
  { label: 'H96 mixed varlen', detail: '6 seqs', cute: 3.24, tirx: 3.03 },
  { label: 'H96 uniform varlen', detail: '8 × 1024', cute: 2.59, tirx: 2.47 },
  { label: 'H64 fixed', detail: '1 × 8192', cute: 2.51, tirx: 3.61 },
  { label: 'H64 mixed varlen', detail: '6 seqs', cute: 3.55, tirx: 3.29 },
  { label: 'H64 uniform varlen', detail: '8 × 1024', cute: 2.56, tirx: 2.43 },
] as const;

type KernelFilter = 'both' | 'cute' | 'tirx';

export function SpeedupChart() {
  const [filter, setFilter] = useState<KernelFilter>('both');
  const scaleMax = 4;
  const rows = [...workloads, { label: 'Geomean', detail: 'all six', cute: 2.85, tirx: 2.96 }];

  return (
    <Reveal className="chart-card chart-dark speedup-chart" label="Speedup over FlashKDA on B300 for six workloads. Geomean: KDA + CAKE 2.85×, KDA + TIRx 2.96×.">
      <div className="chart-head">
        <span>SPEEDUP VS. FLASHKDA · B300 · 8,192 TOKENS</span>
        <div className="chart-toggle" role="group" aria-label="Show kernels">
          {(['both', 'cute', 'tirx'] as const).map((value) => (
            <button key={value} type="button" aria-pressed={filter === value} onClick={() => setFilter(value)}>
              {value === 'both' ? 'Both' : value === 'cute' ? 'KDA + CAKE' : 'KDA + TIRx'}
            </button>
          ))}
        </div>
      </div>
      <div className="speedup-rows">
        <div className="speedup-axis" aria-hidden="true">
          {[0, 1, 2, 3, 4].map((tick) => (
            <span key={tick} style={{ left: `${(tick / scaleMax) * 100}%` }}>{tick}×</span>
          ))}
          <i className="speedup-baseline" style={{ left: `${100 / scaleMax}%` }}><b>FlashKDA 1.00×</b></i>
        </div>
        {rows.map((row, index) => (
          <div className={`speedup-row${row.label === 'Geomean' ? ' is-total' : ''}`} key={row.label}>
            <div className="speedup-label">
              <strong>{row.label}</strong>
              <span>{row.detail}</span>
            </div>
            <div className="speedup-bars">
              {(['cute', 'tirx'] as const).map((kernel) => (
                <div
                  key={kernel}
                  className={`speedup-bar bar-${kernel}${filter !== 'both' && filter !== kernel ? ' is-muted' : ''}`}
                  style={{ '--w': row[kernel] / scaleMax, '--i': index } as CSSProperties}
                >
                  <i />
                  <span>{row[kernel].toFixed(2)}×</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="chart-legend">
        <span><i className="swatch swatch-cute" /> KDA + CAKE</span>
        <span><i className="swatch swatch-tirx" /> KDA + TIRx</span>
        <span><i className="swatch swatch-base" /> FlashKDA forward baseline</span>
      </div>
    </Reveal>
  );
}

export function AccuracyChart() {
  const width = 640;
  const height = 300;
  const left = 46;
  const right = 626;
  const top = 18;
  const bottom = 266;
  const x = (value: number) => left + (value / 8192) * (right - left);
  const y = (value: number) => bottom - (value / 0.7) * (bottom - top);
  const series = [
    { key: 'base', name: 'FlashKDA', data: accuracyFlashKda },
    { key: 'tirx', name: 'Ours (TIRx)', data: accuracyTirx },
    { key: 'cute', name: 'Ours (CuTe)', data: accuracyCute },
  ] as const;
  const finalState = [
    { key: 'base', name: 'FlashKDA', value: 3.45 },
    { key: 'cute', name: 'CuTe', value: 0.22 },
    { key: 'tirx', name: 'TIRx', value: 0.29 },
  ] as const;

  return (
    <Reveal
      className="chart-card chart-dark accuracy-chart"
      threshold={0.35}
      label="Output relative RMSE versus context length for a Kimi-Linear-48B prefill of 8,183 tokens: FlashKDA drifts from 0.43% to about 0.69%, while both of our kernels stay between 0.23% and 0.43%. Final-state relative RMSE: FlashKDA 3.45%, CuTe 0.22%, TIRx 0.29%."
    >
      <div className="chart-head">
        <span>KIMI-LINEAR-48B PREFILL · MATH-500 PROMPT · 96 HEADS</span>
        <span className="chart-live"><i /> CONTEXT <CountUp to={8183} duration={2600} linear /> TOKENS</span>
      </div>
      <span className="chart-scroll-hint">Swipe to explore the full plots <span aria-hidden="true">→</span></span>
      <div className="accuracy-grid">
        <div className="accuracy-plot">
          <p className="plot-title">Output error vs. context length <span>relative RMSE, %</span></p>
          <div className="plot-scroll" role="region" aria-label="Output error plot, scroll horizontally for full detail" tabIndex={0}>
            <svg viewBox={`0 0 ${width} ${height}`} focusable="false" aria-hidden="true">
              {[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7].map((tick) => (
                <g key={tick}>
                  <path className="grid-line" d={`M${left} ${y(tick)}H${right}`} />
                  <text className="axis-text" x={left - 10} y={y(tick) + 3} textAnchor="end">{tick.toFixed(1)}</text>
                </g>
              ))}
              {[0, 2048, 4096, 6144, 8192].map((tick) => (
                <text key={tick} className="axis-text" x={x(tick)} y={bottom + 20} textAnchor="middle">{tick ? `${tick / 1024}k` : '0'}</text>
              ))}
              {series.map(({ key, data }) => (
                <path key={key} className={`plot-line line-${key}`} d={linePath(data, x, y)} pathLength={1} />
              ))}
              <g className="plot-cursor">
                <path d={`M${left} ${top}V${bottom}`} />
              </g>
            </svg>
          </div>
          <div className="chart-legend">
            {series.map(({ key, name }) => (
              <span key={key}><i className={`swatch swatch-${key}`} /> {name}</span>
            ))}
          </div>
        </div>
        <div className="accuracy-final">
          <p className="plot-title">Final state after 8,183 tokens <span>relative RMSE, %</span></p>
          <div className="final-bars">
            {finalState.map(({ key, name, value }) => (
              <div className={`final-bar final-${key}`} key={key} style={{ '--h': value / 3.6 } as CSSProperties}>
                <strong><CountUp to={value} decimals={2} suffix="%" delay={2500} duration={1100} /></strong>
                <i />
                <span>{name}</span>
              </div>
            ))}
          </div>
          <p className="final-callout"><b>~1/10</b> of FlashKDA&apos;s final-state error, close to FLA</p>
        </div>
      </div>
    </Reveal>
  );
}

const humanizeModels = [
  { name: 'GPT-5.6-sol', putnam: [3, 46, 50], physics: [31, 41, 44] },
  { name: 'Kimi-K3', putnam: [1, 4, 47], physics: [35, 37, 42] },
  { name: 'GLM-5.3', putnam: [0, 2, 25], physics: [24, 34, 40] },
  { name: 'DeepSeek V4 Pro', putnam: [0, 4, 13], physics: [32, 35, 39] },
] as const;

const scaffoldLevels = ['Model (API)', 'Tool (CLI)', 'Flow (Humanize)'] as const;

export function HumanizeChart() {
  const [bench, setBench] = useState<'putnam' | 'physics'>('putnam');

  return (
    <Reveal className="chart-card chart-light humanize-chart" label="Humanize ablation. PutnamBench: GPT-5.6-sol 3 / 46 / 50, Kimi-K3 1 / 4 / 47, GLM-5.3 0 / 2 / 25, DeepSeek V4 Pro 0 / 4 / 13. Physics Cup: GPT-5.6-sol 31 / 41 / 44, Kimi-K3 35 / 37 / 42, GLM-5.3 24 / 34 / 40, DeepSeek V4 Pro 32 / 35 / 39 for API, CLI, and Humanize respectively.">
      <div className="chart-head">
        <span>HUMANIZE ABLATION · SAME MODEL, THREE LEVELS OF SCAFFOLDING</span>
        <div className="chart-toggle" role="group" aria-label="Benchmark">
          <button type="button" aria-pressed={bench === 'putnam'} onClick={() => setBench('putnam')}>PutnamBench</button>
          <button type="button" aria-pressed={bench === 'physics'} onClick={() => setBench('physics')}>Physics Cup</button>
        </div>
      </div>
      <div className="humanize-groups">
        {humanizeModels.map((model, group) => {
          const values = model[bench];
          return (
            <div className="humanize-group" key={model.name}>
              <div className="humanize-bars">
                {values.map((value, level) => (
                  <div
                    className={`humanize-bar level-${level}`}
                    key={scaffoldLevels[level]}
                    style={{ '--h': value / 50, '--i': group * 3 + level } as CSSProperties}
                  >
                    <span>
                      {value}
                      {level > 0 && <small>+{value - values[0]}</small>}
                    </span>
                    <i />
                  </div>
                ))}
              </div>
              <strong>{model.name}</strong>
            </div>
          );
        })}
      </div>
      <div className="chart-legend">
        {scaffoldLevels.map((level, index) => (
          <span key={level}><i className={`swatch swatch-level-${index}`} /> {level}</span>
        ))}
        <span className="legend-note">Green deltas are gains over the raw API.</span>
      </div>
    </Reveal>
  );
}

export function DecayMeter() {
  const max = 650;
  const markers = [
    { bits: 52, title: '≈52 bits', copy: 'deepest decay in the random tests' },
    { bits: 126, title: '126 bits', copy: 'hack D’s denominator underflows to 0' },
    { bits: 600, title: '≈600 bits', copy: 'p99 decay per 64 tokens in real Kimi-Linear' },
  ] as const;

  return (
    <Reveal className="decay-meter" threshold={0.45} label="Gate decay depth per 64 tokens. Random tests reach about 52 bits, the hacked TIRx kernel underflows beyond 126 bits, and real Kimi-Linear gates reach about 600 bits at p99.">
      <div className="chart-head">
        <span>CUMULATIVE GATE DECAY WITHIN ONE 64-TOKEN CHUNK</span>
        <span className="chart-live"><i /> <CountUp to={600} duration={3400} linear /> BITS</span>
      </div>
      <div className="decay-track">
        <div className="decay-danger" style={{ left: `${(126 / max) * 100}%` }} aria-hidden="true" />
        <div className="decay-fill" style={{ '--w': 600 / max } as CSSProperties} aria-hidden="true" />
        {markers.map(({ bits, title, copy }, index) => (
          <div className={`decay-marker marker-${index}`} key={bits} style={{ left: `${(bits / max) * 100}%` }}>
            <i />
            <strong>{title}</strong>
            <span>{copy}</span>
          </div>
        ))}
      </div>
      <div className="decay-status" aria-hidden="true">
        <span className="decay-status-label">HACK D ON THIS INPUT</span>
        <span className="decay-ok">output finite · test passes · 3.57×</span>
        <span className="decay-nan">denominator = 0 · output NaN</span>
      </div>
    </Reveal>
  );
}

export function AblationCharts() {
  const width = 520;
  const height = 280;
  const left = 44;
  const right = 500;
  const top = 16;
  const bottom = 244;
  const x = (hours: number) => left + (hours / 16) * (right - left);
  const ySpeed = (value: number) => bottom - ((value - 0.4) / 1.6) * (bottom - top);
  const yTokens = (value: number) => bottom - (value / 3) * (bottom - top);
  const hourTicks = [0, 4, 8, 12, 16];
  const axes = (ticks: readonly number[], y: (value: number) => number, label: (value: number) => string) => (
    <>
      {ticks.map((tick) => (
        <g key={tick}>
          <path className="grid-line" d={`M${left} ${y(tick)}H${right}`} />
          <text className="axis-text" x={left - 9} y={y(tick) + 3} textAnchor="end">{label(tick)}</text>
        </g>
      ))}
      {hourTicks.map((tick) => (
        <text key={tick} className="axis-text" x={x(tick)} y={bottom + 20} textAnchor="middle">{tick}h</text>
      ))}
    </>
  );
  const last = <T extends Series>(series: T) => series[series.length - 1];

  return (
    <Reveal className="chart-card chart-dark ablation-chart" threshold={0.3} label="Speedup and output tokens over a 14-hour budget. Focusing on one simple shape reaches 1.85× and 2.74 million output tokens; targeting all six shapes at once reaches 1.01× and 1.67 million output tokens.">
      <div className="chart-head">
        <span>SAME B300 · SAME 14-HOUR BUDGET · TWO STRATEGIES</span>
        <span className="chart-live"><i /> T + <CountUp to={14} duration={3000} linear /> H</span>
      </div>
      <span className="chart-scroll-hint">Swipe to explore the full plots <span aria-hidden="true">→</span></span>
      <div className="ablation-grid">
        <div>
          <p className="plot-title">Best speedup on the evaluation shape</p>
          <div className="plot-scroll" role="region" aria-label="Speedup plot, scroll horizontally for full detail" tabIndex={0}>
            <svg viewBox={`0 0 ${width} ${height}`} focusable="false" aria-hidden="true">
              {axes([0.4, 0.8, 1.2, 1.6, 2.0], ySpeed, (tick) => `${tick.toFixed(1)}×`)}
              <path className="baseline-line" d={`M${left} ${ySpeed(1)}H${right}`} />
              <text className="axis-text baseline-text" x={left + 8} y={ySpeed(1) - 7}>FlashKDA = 1.0×</text>
              <path className="plot-line line-focus step-line" d={stepPath(speedupFocused, x, ySpeed)} pathLength={1} />
              <path className="plot-line line-all step-line" d={stepPath(speedupAllShapes, x, ySpeed)} pathLength={1} />
              {speedupFocused.map(([hours, value]) => (
                <circle key={`f${hours}`} className="plot-dot dot-focus" cx={x(hours)} cy={ySpeed(value)} r={3.4} style={{ '--t': hours / 14 } as CSSProperties} />
              ))}
              {speedupAllShapes.map(([hours, value]) => (
                <circle key={`a${hours}`} className="plot-dot dot-all" cx={x(hours)} cy={ySpeed(value)} r={3.4} style={{ '--t': hours / 14 } as CSSProperties} />
              ))}
              <text className="end-label label-focus" x={x(last(speedupFocused)[0]) + 10} y={ySpeed(last(speedupFocused)[1]) + 4}>1.85×</text>
              <text className="end-label label-all" x={x(last(speedupAllShapes)[0]) + 10} y={ySpeed(last(speedupAllShapes)[1]) + 4}>1.01×</text>
            </svg>
          </div>
        </div>
        <div>
          <p className="plot-title">Cumulative output tokens</p>
          <div className="plot-scroll" role="region" aria-label="Output tokens plot, scroll horizontally for full detail" tabIndex={0}>
            <svg viewBox={`0 0 ${width} ${height}`} focusable="false" aria-hidden="true">
              {axes([0, 1, 2, 3], yTokens, (tick) => `${tick}M`)}
              <path className="plot-line line-focus" d={linePath(tokensFocused, x, yTokens)} pathLength={1} />
              <path className="plot-line line-all" d={linePath(tokensAllShapes, x, yTokens)} pathLength={1} />
              <text className="end-label label-focus" x={x(last(tokensFocused)[0]) + 8} y={yTokens(last(tokensFocused)[1]) + 4}>2.74M</text>
              <text className="end-label label-all" x={x(last(tokensAllShapes)[0]) + 8} y={yTokens(last(tokensAllShapes)[1]) + 4}>1.67M</text>
            </svg>
          </div>
        </div>
      </div>
      <div className="chart-legend">
        <span><i className="swatch swatch-focus" /> One simple shape first</span>
        <span><i className="swatch swatch-all" /> All six shapes at once</span>
      </div>
    </Reveal>
  );
}

const loopStats = [
  {
    key: 'focused',
    title: 'One simple shape first',
    latency: 0.9,
    stats: [
      { value: 1.85, decimals: 2, suffix: '×', label: 'final speedup' },
      { value: 248, decimals: 0, suffix: '', label: 'hardware tests' },
      { value: 120, decimals: 0, suffix: '', label: 'commits' },
      { value: 2.74, decimals: 2, suffix: 'M', label: 'output tokens' },
    ],
  },
  {
    key: 'all',
    title: 'All six shapes at once',
    latency: 1.9,
    stats: [
      { value: 1.01, decimals: 2, suffix: '×', label: 'final speedup' },
      { value: 159, decimals: 0, suffix: '', label: 'hardware tests' },
      { value: 22, decimals: 0, suffix: '', label: 'commits' },
      { value: 1.67, decimals: 2, suffix: 'M', label: 'output tokens' },
    ],
  },
] as const;

export function LoopDuel() {
  return (
    <Reveal className="loop-duel" threshold={0.35} label="Feedback loops: one simple shape takes a median 0.9 minutes per test and completes 248 tests and 120 commits; all six shapes take 1.9 minutes per test and complete 159 tests and 22 commits.">
      {loopStats.map(({ key, title, latency, stats }) => (
        <article className={`loop-card loop-${key}`} key={key}>
          <div className="loop-ring" style={{ '--period': `${latency * 2}s` } as CSSProperties} aria-hidden="true">
            <svg viewBox="0 0 120 120" focusable="false">
              <circle className="loop-track" cx="60" cy="60" r="46" />
              <g className="loop-spinner">
                <circle className="loop-packet" cx="60" cy="14" r="6" />
              </g>
            </svg>
            <div>
              <strong><CountUp to={latency} decimals={1} duration={900} /></strong>
              <span>min / test</span>
            </div>
          </div>
          <div className="loop-copy">
            <h3>{title}</h3>
            <dl>
              {stats.map(({ value, decimals, suffix, label }, index) => (
                <div key={label}>
                  <dt>{label}</dt>
                  <dd><CountUp to={value} decimals={decimals} suffix={suffix} duration={1500} delay={index * 120} /></dd>
                </div>
              ))}
            </dl>
          </div>
        </article>
      ))}
    </Reveal>
  );
}
