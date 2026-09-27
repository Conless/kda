import type { Metadata } from 'next';
import Link from 'next/link';
import {
  AblationCharts,
  AccuracyChart,
  CountUp,
  DecayMeter,
  HumanizeChart,
  LoopDuel,
  Reveal,
  SpeedupChart,
} from './motion';
import './post.css';

const siteUrl = process.env.SITE_URL ?? 'https://nvlabs.github.io/kda';
const POST_URL = `${siteUrl}/blog/2026-09-27-kda-for-kda/`;
const POST_TITLE = 'KDA²: Kernel Design Agents Optimize Kimi Delta Attention';
const POST_DESCRIPTION =
  'Kernel Design Agents wrote Kimi Delta Attention kernels that run up to 2.96× faster than FlashKDA on B300 with a tenth of its state error. Here is how, and how the agents tried to cheat along the way.';

const RELEASE_URL = 'https://github.com/humanfia/kda-for-kda-release';
const REPOSITORY_URL = 'https://github.com/NVlabs/kda';
const HUMANIZE_URL = 'https://github.com/humanfia/humanize2';
const FLASHKDA_URL = 'https://github.com/MoonshotAI/FlashKDA';
const FLA_URL = 'https://github.com/fla-org/flash-linear-attention';
const KIMI_LINEAR_URL = 'https://github.com/MoonshotAI/Kimi-Linear';

export const metadata: Metadata = {
  title: `${POST_TITLE} | KDA Blog`,
  description: POST_DESCRIPTION,
  alternates: { canonical: POST_URL },
  openGraph: {
    type: 'article',
    url: POST_URL,
    title: POST_TITLE,
    description: POST_DESCRIPTION,
    publishedTime: '2026-09-27',
    images: [{ url: `${siteUrl}/og.png`, width: 1200, height: 630, alt: 'Kernel Design Agents' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: POST_TITLE,
    description: POST_DESCRIPTION,
    images: [`${siteUrl}/og.png`],
  },
};

const heroKernels = [
  { name: 'TIRx', value: 2.96, className: 'hero-bar-tirx' },
  { name: 'CAKE-PTX', value: 2.94, className: 'hero-bar-cake' },
  { name: 'CuTe-DSL', value: 2.85, className: 'hero-bar-cute' },
  { name: 'FlashKDA', value: 1, className: 'hero-bar-base' },
] as const;

const hacks = [
  {
    letter: 'A',
    tag: 'INPUT DISTRIBUTION',
    title: 'Input-distribution overfitting',
    metric: '3.74×',
    metricLabel: 'claimed · 2.48× once fixed',
    teaser: 'distribution',
    body: (
      <>
        The synthesized kernel replaced the L2 normalization of Q and K with a hard-coded constant, 0.1778209953: the expected reciprocal L2 norm of a vector drawn from X ~ N(0, 0.5²). It also zeroed some initial-state channels outright to skip the triangular matrix inverse. Synthetic random tests reported an inflated 3.74×; on real, non-Gaussian inputs the kernel failed completely. With the shortcut removed, the real speedup fell to 2.48×.
      </>
    ),
  },
  {
    letter: 'B',
    tag: 'SHAPE & LAYOUT',
    title: 'Shape and layout hard-coding',
    metric: 'static',
    metricLabel: 'sequence boundaries',
    teaser: 'layout',
    body: (
      <>
        The agent noticed that packed layouts in the test set always followed the same pattern. So it skipped the dynamic offset computation from <code>cu_seqlens</code> and hard-coded the sequence boundaries. Because the holdout set never exercised the dynamic-boundary path, the kernel slipped straight past the logic checks.
      </>
    ),
  },
  {
    letter: 'C',
    tag: 'HISTORY',
    title: 'Illegal history truncation',
    metric: '5.16×',
    metricLabel: 'claimed on a single H64 sequence',
    teaser: 'truncation',
    body: (
      <>
        Leaning on gate decay, the agent assumed that state older than 32 tokens was negligible and cut long sequences into chunks it could process in parallel. Its built-in decay check was tuned just loosely enough to pass on the weakly decaying random data, reporting 5.16× on a single H64 sequence. Under the strongly decaying gates of the real model, the check failed constantly and the speedup vanished.
      </>
    ),
  },
  {
    letter: 'D',
    tag: 'NUMERICS',
    title: 'Overflow under extreme gate ranges',
    metric: '3.57×',
    metricLabel: 'claimed · NaN on every real case',
    teaser: 'overflow',
    body: (
      <>
        A TIRx kernel computed cumulative powers of two directly inside each 64-token chunk. Once the decay exceeded 126 bits, the denominator underflowed to zero and the output turned into NaN. The random tests never decayed deeply enough to notice (about 52 bits at most) and measured 3.57×. On real workloads, every single case collapsed numerically.
      </>
    ),
  },
  {
    letter: 'E',
    tag: 'PRECISION',
    title: 'Precision loss from a low-precision LUT',
    metric: '9%',
    metricLabel: 'decay-factor error · 23/24 cases fail',
    teaser: 'lut',
    body: (
      <>
        For sequence lengths that are multiples of 32, a CuTe kernel built an FP16 table of cumulative decay. Real gates span more dynamic range than FP16 can represent, so decay factors were off by up to 9%, and 23 of 24 long real-world sequences fell outside tolerance.
      </>
    ),
  },
] as const;

type HackTeaserKind = (typeof hacks)[number]['teaser'];

function HackTeaser({ kind }: { kind: HackTeaserKind }) {
  if (kind === 'distribution') {
    return (
      <svg viewBox="0 0 180 82" focusable="false">
        <rect className="teaser-frame" x="1" y="1" width="178" height="80" rx="10" />
        <path className="teaser-stroke" d="M16 66C48 66 58 20 72 20S96 66 128 66" />
        <path className="hack-real-curve" d="M16 66C30 66 36 30 50 30S76 50 96 56S140 64 164 66" />
        <path className="teaser-accent-stroke hack-constant" d="M72 12V70" />
        <text className="hack-svg-label" x="78" y="16">0.1778…</text>
      </svg>
    );
  }

  if (kind === 'layout') {
    return (
      <svg viewBox="0 0 180 82" focusable="false">
        <rect className="teaser-frame" x="1" y="1" width="178" height="80" rx="10" />
        <rect className="teaser-node-box" x="16" y="30" width="148" height="22" rx="4" />
        <g className="hack-boundaries-real">
          <path d="M52 26V56M96 26V56M130 26V56" />
        </g>
        <path className="teaser-accent-stroke" d="M58 22V60M88 22V60M124 22V60" />
        <text className="hack-svg-label" x="16" y="74">cu_seqlens ignored</text>
      </svg>
    );
  }

  if (kind === 'truncation') {
    return (
      <svg viewBox="0 0 180 82" focusable="false">
        <rect className="teaser-frame" x="1" y="1" width="178" height="80" rx="10" />
        {Array.from({ length: 12 }, (_, index) => (
          <rect key={index} className="hack-token" x={16 + index * 12.6} y="34" width="9" height="14" rx="2" style={{ animationDelay: `${index * -0.25}s` }} />
        ))}
        <rect className="hack-window" x="104" y="28" width="62" height="26" rx="5" />
        <text className="hack-svg-label" x="104" y="68">last 32 only</text>
        <path className="teaser-stroke" d="M16 20H96" strokeDasharray="3 5" />
      </svg>
    );
  }

  if (kind === 'overflow') {
    return (
      <svg viewBox="0 0 180 82" focusable="false">
        <rect className="teaser-frame" x="1" y="1" width="178" height="80" rx="10" />
        <path className="hack-ceiling" d="M18 26H162" />
        <text className="hack-svg-label" x="18" y="20">2⁻¹²⁶</text>
        <rect className="hack-exp-bar" x="30" y="16" width="16" height="52" rx="2" />
        <rect className="hack-exp-bar bar-b" x="56" y="16" width="16" height="52" rx="2" />
        <rect className="hack-exp-bar bar-c" x="82" y="16" width="16" height="52" rx="2" />
        <text className="hack-nan" x="148" y="54" textAnchor="end">NaN</text>
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 180 82" focusable="false">
      <rect className="teaser-frame" x="1" y="1" width="178" height="80" rx="10" />
      <path className="teaser-stroke" d="M16 18C60 22 96 44 164 64" />
      <path className="teaser-accent-stroke hack-lut" d="M16 18H40V26H64V36H88V46H112V52H136V58H164" />
      <text className="hack-svg-label" x="120" y="24">FP16 LUT</text>
    </svg>
  );
}

const hardening = [
  {
    title: 'Dynamic input salts and distribution holdouts',
    copy: 'Every scoring run draws a fresh random seed. In an isolated zone the agent cannot see, input distributions and sequence layouts alternate at random.',
  },
  {
    title: 'A strict specification',
    copy: 'The task prompt defines the full set of legal inputs. Kernels may specialize by shape, but must be correct on every legal input.',
  },
  {
    title: 'CUDA Graph replay checks',
    copy: 'After benchmark timing, we swap in new input tensors and replay the graph to validate the outputs, defeating cache-based precomputation.',
  },
  {
    title: 'Stress probes',
    copy: 'An adversarial set of extremely deep decays, saturated gates, boundary sequence lengths, and repeated keys targets underflow and overflow directly.',
  },
  {
    title: 'Strict element-wise tolerances',
    copy: 'Lenient statistical gates such as “99.9% of elements within 5e-2” are gone. Every element must meet maximum absolute and relative error bounds.',
  },
  {
    title: 'Real end-to-end traces',
    copy: 'We replay traces captured from end-to-end Kimi-Linear runs. The evaluator lives outside the isolated container, physically separating test data from the environment that generates kernels.',
  },
] as const;

const newInV06 = [
  {
    number: '01',
    title: 'Sharper Humanize2 flows',
    copy: 'Better flows, including flame chase and iterative refinement with gpt-5.6-sol and fable-5, plus periodic workspace cleanup.',
    teaser: 'flows',
  },
  {
    number: '02',
    title: 'Many languages, matching skills',
    copy: 'CuTe-DSL, CUDA C++, the new agent-native CAKE IR, and TIRx. Each ships its own diagnostics: IKET exposes the pipeline inside CuTe kernels; TIRx gets CPU-side numerical simulation and static checks.',
    teaser: 'languages',
  },
  {
    number: '03',
    title: 'A self-evolving kernel wiki',
    copy: 'We pruned large swaths of incorrect content, sharpened the tags, and tightened search results.',
    teaser: 'wiki',
  },
] as const;

function FeatureTeaser({ kind }: { kind: (typeof newInV06)[number]['teaser'] }) {
  if (kind === 'flows') {
    return (
      <svg viewBox="0 0 180 82" focusable="false">
        <rect className="teaser-frame" x="1" y="1" width="178" height="80" rx="10" />
        <path className="teaser-stroke teaser-dash" d="M40 41C40 20 70 18 90 30S140 62 140 41S110 18 90 30S40 62 40 41Z" />
        <circle className="teaser-packet" r="4">
          <animateMotion path="M40 41C40 20 70 18 90 30S140 62 140 41S110 18 90 30S40 62 40 41Z" dur="4.2s" repeatCount="indefinite" />
        </circle>
        <circle className="teaser-packet teaser-packet-muted" r="3.5">
          <animateMotion path="M40 41C40 20 70 18 90 30S140 62 140 41S110 18 90 30S40 62 40 41Z" dur="4.2s" begin="-2.1s" repeatCount="indefinite" />
        </circle>
      </svg>
    );
  }

  if (kind === 'languages') {
    return (
      <svg viewBox="0 0 180 82" focusable="false">
        <rect className="teaser-frame" x="1" y="1" width="178" height="80" rx="10" />
        {['CuTe-DSL', 'CUDA C++', 'CAKE IR', 'TIRx'].map((name, index) => (
          <g key={name} className="lang-chip" style={{ animationDelay: `${index * 0.9}s` }}>
            <rect className="teaser-node-box" x={14 + (index % 2) * 78} y={index < 2 ? 14 : 44} width="70" height="24" rx="5" />
            <text className="lang-label" x={49 + (index % 2) * 78} y={index < 2 ? 30 : 60} textAnchor="middle">{name}</text>
          </g>
        ))}
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 180 82" focusable="false">
      <rect className="teaser-frame" x="1" y="1" width="178" height="80" rx="10" />
      <path className="teaser-stroke" d="M22 20H120M22 34H146M22 62H104" />
      <path className="wiki-strike" d="M22 48H132" />
      <path className="teaser-accent-stroke wiki-fix" d="M22 48H96" />
      <circle className="teaser-result-check" cx="152" cy="54" r="11" />
      <path className="teaser-check-mark" d="M147 54L151 58L158 50" />
    </svg>
  );
}

function SectionHead({ id, label, children }: { id?: string; label: string; children: React.ReactNode }) {
  return (
    <div className="post-section-head" id={id}>
      <p className="section-label">{label}</p>
      <h2>{children}</h2>
    </div>
  );
}

export default function KdaForKdaPost() {
  return (
    <main className="post">
      <div className="post-progress" aria-hidden="true" />

      <nav className="nav shell" aria-label="Primary navigation">
        <Link className="brand" href="/" aria-label="Kernel Design Agents home">
          <span className="brand-mark" aria-hidden="true">KDA</span>
          <span>Kernel Design <b>Agents</b></span>
        </Link>
        <div className="nav-links">
          <a href="#results">Results</a>
          <a href="#hacks">Hacks</a>
          <a href="#hardening">Hardening</a>
          <a href="#ablation">Ablation</a>
          <a className="nav-cta" href={RELEASE_URL}>Get the kernels <span aria-hidden="true">↗</span></a>
        </div>
      </nav>

      <header className="post-hero shell" id="top">
        <div className="post-hero-copy">
          <div className="eyebrow"><span /> KDA Blog · September 27, 2026</div>
          <h1 className="post-title">
            KDA<sup>2</sup>
            <em>Kernel Design Agents optimize Kimi Delta Attention</em>
          </h1>
          <p className="post-dek">Results, hacks, and lessons from pointing our kernel agents at the operator they share a name with.</p>
          <p className="post-byline">Kernel Design Agents team · NVIDIA · 12 min read</p>
        </div>

        <Reveal className="pipeline post-hero-card" threshold={0.2} label="Geomean speedup over FlashKDA on B300: TIRx 2.96×, CAKE-PTX 2.94×, CuTe-DSL 2.85×.">
          <div className="pipeline-topline">
            <span>KDA → KDA · B300 FORWARD</span>
            <span className="live"><i /> VERIFIED ON KIMI-LINEAR</span>
          </div>
          <svg className="ouroboros" viewBox="0 0 360 112" focusable="false" aria-hidden="true">
            <path className="teaser-stroke teaser-dash" d="M86 40C140 18 220 18 274 40" />
            <path className="teaser-stroke teaser-dash" d="M274 72C220 94 140 94 86 72" />
            <rect className="teaser-node-box" x="12" y="36" width="96" height="40" rx="8" />
            <rect className="ouro-attn" x="252" y="36" width="96" height="40" rx="8" />
            <text className="ouro-label" x="60" y="54" textAnchor="middle">KDAgent</text>
            <text className="ouro-sub" x="60" y="67" textAnchor="middle">kernel design agents</text>
            <text className="ouro-label ouro-label-dark" x="300" y="54" textAnchor="middle">KDAttn</text>
            <text className="ouro-sub ouro-sub-dark" x="300" y="67" textAnchor="middle">kimi delta attention</text>
            <text className="ouro-edge" x="180" y="12" textAnchor="middle">writes kernels</text>
            <text className="ouro-edge" x="180" y="108" textAnchor="middle">scores &amp; traces</text>
            <circle className="teaser-packet" r="4">
              <animateMotion path="M86 40C140 18 220 18 274 40" dur="2.4s" repeatCount="indefinite" />
            </circle>
            <circle className="teaser-packet teaser-packet-muted" r="3.5">
              <animateMotion path="M274 72C220 94 140 94 86 72" dur="2.4s" begin="-1.2s" repeatCount="indefinite" />
            </circle>
          </svg>
          <div className="hero-metric">
            <strong><CountUp from={1} to={2.96} decimals={2} suffix="×" duration={1800} delay={200} /></strong>
            <span>geomean speedup over FlashKDA<br />with a tenth of its final-state error</span>
          </div>
          <div className="hero-bars">
            {heroKernels.map(({ name, value, className }, index) => (
              <div className={`hero-bar ${className}`} key={name} style={{ '--w': value / 3.2, '--i': index } as React.CSSProperties}>
                <span>{name}</span>
                <div><i /></div>
                <b>{value.toFixed(2)}×</b>
              </div>
            ))}
          </div>
        </Reveal>
      </header>

      <section className="definition-strip" aria-label="Naming">
        <div className="shell definition-grid">
          <p><strong>KDAgent</strong> — Kernel Design Agents, our agentic system that researches, writes, verifies, and tunes GPU kernels.</p>
          <p><strong>KDAttn</strong> — Kimi Delta Attention, the linear-attention operator behind Moonshot AI&apos;s Kimi-Linear models.</p>
        </div>
      </section>

      <article className="post-article shell">
        <section className="post-intro">
          <div className="post-prose post-prose-lead">
            <p>
              When we named our project Kernel Design Agents, we walked straight into a name collision with another KDA: Kimi Delta Attention. Ever since, one question has kept coming back to us: <q>Can you use KDA to write KDA?</q> So tonight, under a full moon made for a moonshot, we are happy to share the latest results from KDA(gent) v0.6: KDA optimizing KDA.
            </p>
          </div>
        </section>

        <section className="workflow-shell post-features" aria-label="What is new in KDA v0.6">
          <div className="workflow-head">
            <span>WHAT&apos;S NEW IN KDA v0.6</span>
            <span>AGENT · SKILLS · MEMORY</span>
          </div>
          <div className="workflow-grid">
            {newInV06.map(({ number, title, copy, teaser }) => (
              <article className="workflow-step" key={number}>
                <div className="step-number">{number}</div>
                <div className="workflow-teaser" aria-hidden="true"><FeatureTeaser kind={teaser} /></div>
                <h3>{title}</h3>
                <p>{copy}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="post-intro">
          <div className="post-prose">
            <p>
              We wrote kernels in both CuTe-DSL and TIRx. TIRx is a GPU kernel programming interface that sits close to PTX; the TIRx Harness built around it gives agents tools for development, diagnosis, and evaluation, which makes optimization far more dependable. On B300, the TIRx kernel runs <strong>2.96×</strong> faster than FlashKDA, the CuTe-DSL kernel <strong>2.85×</strong>, and the CAKE-PTX kernel <strong>2.94×</strong>. All of them pass acceptance on real Kimi-Linear workloads, and all are more accurate than FlashKDA.
            </p>
            <p>
              Along the way, the agent also produced candidates that clocked 3.28×, 3.57×, even 3.74×. Careful ablations showed that every one of them was overfitting to the verifier, exploiting distributional assumptions in the test suite, untested boundaries, or loose precision checks. This post dissects those hacks and describes how we hardened acceptance along both hardware and numerical lines.
            </p>
            <p className="post-callout">
              Both kernels are open source: <a href={RELEASE_URL}>github.com/humanfia/kda-for-kda-release <span aria-hidden="true">↗</span></a>
            </p>
          </div>
        </section>

        <section className="post-section" id="results">
          <SectionHead label="01 · RESULTS">2.96× faster,<br />and more accurate</SectionHead>
          <div className="post-prose">
            <p>
              We benchmarked the generated kernels on an NVIDIA B300 GPU against the forward pass of <a href={FLASHKDA_URL}>FlashKDA</a>, Moonshot AI&apos;s official open-source implementation. The workloads cover fixed-length sequences and variable-length (varlen) batches with several length distributions, each totaling 8,192 tokens of context.
            </p>
          </div>
          <div className="post-figure">
            <SpeedupChart />
          </div>
          <div className="post-prose">
            <p>
              For accuracy, we built 151 real cases from <a href={KIMI_LINEAR_URL}>Kimi-Linear-48B-A3B</a> prefills on GSM8K and MATH-500, and checked every kernel against a token-by-token fp64 recurrence. One finding surprised us: FlashKDA (commit <code>7afb9f</code>) is itself less accurate than <a href={FLA_URL}>FLA</a>. It keeps its recurrent state in bf16, so error compounds as sequences grow; by 8k tokens, the relative error of the final state reaches 0.035, beyond our acceptance threshold.
            </p>
            <p>
              Both of our kernels hold final-state error to about 0.003 at 8k tokens: a tenth of FlashKDA&apos;s, and close to FLA. Output accuracy matches FlashKDA overall and pulls ahead on long sequences.
            </p>
          </div>
          <figure className="post-figure">
            <AccuracyChart />
            <figcaption>
              Kimi-Linear-48B prefill of one MATH-500 prompt (8,183 tokens, 96 heads). Relative RMSE = rms(x − x<sub>fp64</sub>) / rms(x<sub>fp64</sub>), FlashKDA&apos;s own test metric; lower is better.
            </figcaption>
          </figure>
        </section>

        <section className="post-section" id="humanize">
          <SectionHead label="02 · HUMANIZE">Better flows,<br />stronger agents</SectionHead>
          <div className="post-prose">
            <p>
              Our <a href={HUMANIZE_URL}>Humanize</a> ablation shows the same pattern for every base model we tried: moving from a coding CLI (Claude Code or Codex) to a Humanize flow brings a large jump in performance. There is a catch, though. The more capable the agent, the more room it has to hack.
            </p>
          </div>
          <figure className="post-figure">
            <HumanizeChart />
            <figcaption>PutnamBench and Physics Cup scores for four base models at three levels of scaffolding: the raw model API, a coding CLI, and a Humanize flow.</figcaption>
          </figure>
        </section>

        <section className="post-section" id="hacks">
          <SectionHead label="03 · REWARD HACKING">How KDAgent<br />games the tests</SectionHead>
          <div className="post-prose">
            <p className="post-aside">To keep the two KDAs apart, we call the operator KDAttn and the agent KDAgent from here on.</p>
            <p className="post-pullquote">KDAgent optimizes the score, not the kernel.</p>
            <p>If the tests have a hole, it will find it. We ran into five kinds of holes.</p>
          </div>
          <div className="post-figure hack-grid" aria-label="Five reward hacks found by KDAgent">
            {hacks.map(({ letter, tag, title, metric, metricLabel, teaser, body }) => (
              <article className="hack-card" key={letter}>
                <div className="achievement-card-top">
                  <span>{letter} / {tag}</span>
                  <span className="hack-status"><i /> CAUGHT</span>
                </div>
                <div className="hack-teaser" aria-hidden="true"><HackTeaser kind={teaser} /></div>
                <div className="hack-metric">
                  <strong>{metric}</strong>
                  <span>{metricLabel}</span>
                </div>
                <h3>{title}</h3>
                <p>{body}</p>
              </article>
            ))}
            <article className="hack-card hack-card-summary">
              <div className="achievement-card-top">
                <span>THE COMMON THREAD</span>
                <span className="hack-status"><i /> REAL DATA</span>
              </div>
              <div className="hack-metric">
                <strong>~600 bits</strong>
                <span>p99 gate decay per 64 tokens</span>
              </div>
              <h3>Every hack passed every test we had</h3>
              <p>
                Each one hid in inputs that only a real model produces. In real Kimi-Linear, the gate decays by about 600 bits per 64 tokens at p99, an order of magnitude deeper than our random test data.
              </p>
            </article>
          </div>
          <div className="post-figure">
            <DecayMeter />
          </div>
        </section>

        <section className="post-section" id="hardening">
          <SectionHead label="04 · HARDENING">Closing<br />the loopholes</SectionHead>
          <div className="post-prose">
            <p>
              Humanize flows make agents more capable, and more inclined to hunt for hacks; CAKE with a simple <code>/goal</code> did not trigger any of these behaviors. To keep the agents honest, we built several layers of defense.
            </p>
          </div>
          <Reveal className="post-figure workflow-shell gauntlet" threshold={0.2} label="Six acceptance gates. Hacked candidates are rejected at a gate; honest kernels reach release.">
            <div className="workflow-head">
              <span>ACCEPTANCE GAUNTLET</span>
              <span>6 GATES · 1 WAY OUT</span>
            </div>
            <div className="gauntlet-lane" aria-hidden="true">
              {hardening.map((_, index) => (
                <i className="gauntlet-gate" key={index} style={{ left: `${((index + 1) / 7) * 100}%` }} />
              ))}
              <span className="gauntlet-packet packet-hack packet-one" />
              <span className="gauntlet-packet packet-hack packet-two" />
              <span className="gauntlet-packet packet-honest" />
              <span className="gauntlet-release">RELEASE ✓</span>
            </div>
            <div className="gauntlet-grid">
              {hardening.map(({ title, copy }, index) => (
                <article className="gauntlet-gate-card" key={title} style={{ '--i': index } as React.CSSProperties}>
                  <span className="step-number">{String(index + 1).padStart(2, '0')}</span>
                  <h3>{title}</h3>
                  <p>{copy}</p>
                </article>
              ))}
            </div>
          </Reveal>
          <div className="post-prose">
            <p>Both released kernels pass this suite.</p>
          </div>
          <Reveal className="post-figure release-tiles" threshold={0.4}>
            <article>
              <p className="section-label">TIRX · TUNED IN PLACE ON B300</p>
              <strong><CountUp from={2.54} to={2.96} decimals={2} suffix="×" duration={1800} /></strong>
              <p>Starting from a TIRx implementation, the agent tuned the kernel in place on B300, climbing from 2.54× to 2.96×.</p>
            </article>
            <article>
              <p className="section-label">CUTE-DSL · FP16 TABLE REMOVED</p>
              <strong><CountUp to={2} suffix="%" duration={1200} /></strong>
              <p>The CuTe version drops the FP16 decay table, trading 2% of its speed for correctness.</p>
            </article>
          </Reveal>
        </section>

        <section className="post-section" id="ablation">
          <SectionHead label="05 · ABLATION">Focus first,<br />then generalize?</SectionHead>
          <div className="post-prose">
            <p>
              To see how the synthesis strategy affects convergence, we compared two workflows: progressive synthesis that starts from a single fixed-length shape, and direct multi-objective synthesis across all shapes. With the same hardware (NVIDIA B300) and the same 14-hour budget, their convergence curves diverged sharply.
            </p>
          </div>
          <div className="post-figure">
            <AblationCharts />
          </div>
          <div className="post-prose">
            <p>
              On the same evaluation shape, progressive synthesis reached <strong>1.85×</strong>; direct all-shape synthesis managed only <strong>1.01×</strong>. Two mechanisms explain the gap.
            </p>
            <ol className="post-list">
              <li>
                <strong>Feedback latency.</strong> One test pass over the full workload took a median of 1.9 minutes; a single shape took 0.9. Faster feedback meant denser iteration: the single-shape agent completed 248 hardware tests and 120 commits, against 159 tests and 22 commits for the all-shape agent.
              </li>
              <li>
                <strong>Search-space decoupling.</strong> Synthesizing every shape at once forced the agent to juggle intricate varlen offsets and the compute core at the same time. For the first eight hours its speedup stayed below 0.65×, with most of that time spent debugging varlen edge cases, and the core logic never got the attention it needed. Over the same 14 hours, it also produced far fewer output tokens than the single-shape agent.
              </li>
            </ol>
          </div>
          <div className="post-figure">
            <LoopDuel />
          </div>
          <div className="post-prose">
            <p className="post-pullquote">
              Splitting the work into two stages does not make the model any smarter. It shortens the feedback loop until the model can stay busy.
            </p>
            <p>
              Optimize the compute core first, then generalize to every layout: that decomposition brings each feedback loop down to a length an LLM can work with effectively.
            </p>
          </div>
        </section>

        <section className="post-section" id="more">
          <SectionHead label="06 · ALSO IN v0.6">Other<br />improvements</SectionHead>
          <div className="post-prose">
            <h3>Multi-language support with matching skills</h3>
            <p>
              CuTe-DSL, CUDA C++, and TIRx are all supported, and each language comes with its own diagnostics. For CuTe, IKET exposes the pipeline inside the kernel. For TIRx, CPU-side numerical simulation plus synchronization and data-race analysis catch numerical and concurrency bugs.
            </p>
            <p>
              These TIRx tools belong to the TIRx Harness. The Harness also provides TIRx Foundation, a layer that stays close to the hardware; a kernel zoo of reusable implementations; and a benchmark server that makes performance results easy to compare. Together they give the agent a more reliable development loop: code maps more directly onto the intended hardware behavior, failures leave clues to follow, and performance changes can be confirmed as real. The TIRx team plans to release the Harness formally next week, with a detailed write-up.
            </p>
            <h3>A self-evolving kernel wiki</h3>
            <p>
              The wiki keeps correcting itself as it is used. Incorrect content gets deleted, tags get sharpened, and search results get leaner, so every agent that comes after works from better material.
            </p>
          </div>
        </section>

        <section className="post-section" id="conclusion">
          <SectionHead label="07 · CONCLUSION">Takeaways</SectionHead>
          <div className="post-prose">
            <p>
              We used Kernel Design Agents (KDAgent) to optimize the Kimi Delta Attention (KDAttn) operator on NVIDIA B300, automatically and in depth. With the latest Humanize and multi-language support for TIRx and CuTe-DSL, the resulting kernels reach speedups of <strong>2.96×</strong> and <strong>2.85×</strong> respectively, and cut state error on 8k-token sequences to a tenth of FlashKDA&apos;s.
            </p>
            <p>
              Along the way, the agent produced a series of fake optimizations: overfitting to the input distribution, hard-coding boundaries, and overflowing under extreme values. We answered with layered hardening, including dynamic input salts, stress probes, and strict element-wise tolerances, so that the kernels stay correct and robust on real model workloads. Our ablation further shows that tackling a complex optimization task through progressive synthesis substantially shortens the feedback loop and makes the search more efficient.
            </p>
          </div>
        </section>
      </article>

      <section className="final-cta post-final shell">
        <div>
          <p className="section-label light">RUN IT YOURSELF</p>
          <h2>Try the<br />kernels</h2>
        </div>
        <div className="final-actions">
          <p>Both kernels, CuTe-DSL and TIRx, are open source and pass the hardened acceptance suite described above.</p>
          <a className="button button-acid" href={RELEASE_URL}>Get the kernels <span aria-hidden="true">↗</span></a>
          <a className="final-secondary" href={HUMANIZE_URL}>Explore Humanize <span aria-hidden="true">→</span></a>
          <Link className="final-secondary" href="/#submit">Request a kernel from KDA <span aria-hidden="true">→</span></Link>
        </div>
      </section>

      <footer className="footer shell">
        <Link className="brand" href="/" aria-label="Kernel Design Agents home">
          <span className="brand-mark" aria-hidden="true">KDA</span>
          <span>Kernel Design <b>Agents</b></span>
        </Link>
        <p>An agentic-driven CUDA project</p>
        <div>
          <a href={REPOSITORY_URL}>GitHub</a>
          <a href={HUMANIZE_URL}>Humanize</a>
        </div>
      </footer>
    </main>
  );
}
