// Values transcribed from the dated table in the accompanying KDA² figure.
// Only the series plotted in the figure are needed for this chart.
const history = [
  { date: '6/16', int21: 1.5 },
  { date: '7/21', kda: 1.61 },
  { date: '7/30', cake: 2.048, kda: 1.61 },
  { date: '8/10', cake: 2.048, kda: 1.61 },
  { date: '8/14', cake: 2.09, kda: 2.45 },
  { date: '8/25', cake: 2.373, kda: 2.45 },
  { date: '8/30', cake: 2.373, kda: 2.54 },
  { date: '9/1', cake: 2.373, kda: 2.56 },
  { date: '9/2', cake: 2.373, kda: 2.93 },
  { date: '9/3', cake: 2.373, kda: 2.93 },
  { date: '9/4', cake: 2.732, kda: 2.93 },
  { date: '9/6', cake: 2.94, kda: 2.94 },
  { date: '9/12', kda: 2.96 },
] as const;

type Method = 'int21' | 'cake' | 'kda';

const left = 50;
const right = 1040;
const top = 108;
const bottom = 336;
const x = (index: number) => left + (index / (history.length - 1)) * (right - left);
const y = (value: number) => bottom - ((value - 1.5) / 1.6) * (bottom - top);

const milestones = [
  { label: 'Humanize2', detail: '8/14 · KDA 2.45×', labelX: 345, tone: 'humanize', arrows: [
    { index: 4, score: 2.45, startX: 380 },
  ] },
  { label: 'TIRx', detail: '8/30 · 9/2 · 9/12', labelX: 570, tone: 'tirx', arrows: [
    { index: 6, score: 2.54, startX: 550 },
    { index: 8, score: 2.93, startX: 600 },
    { index: 12, score: 2.96, startX: 625 },
  ] },
  { label: 'CAKE', detail: '9/1 with KDA · 9/6', labelX: 840, tone: 'cake', arrows: [
    { index: 7, score: 2.56, startX: 795 },
    { index: 11, score: 2.94, startX: 865 },
  ] },
] as const;

function value(row: (typeof history)[number], method: Method): number | undefined {
  return method in row ? (row as Partial<Record<Method, number>>)[method] : undefined;
}

function line(method: 'cake' | 'kda') {
  return history.flatMap((row, index) => {
    const current = value(row, method);
    return current === undefined ? [] : [`${index === (method === 'kda' ? 1 : 2) ? 'M' : 'L'}${x(index).toFixed(1)} ${y(current).toFixed(1)}`];
  }).join(' ');
}

export function ProgressTimeline() {
  return (
    <figure className="post-figure progress-figure">
      <div className="progress-card">
        <div className="progress-heading">
          <div>
            <span className="section-label">HOW THE RESULT EVOLVED · 2026</span>
            <h3>From 1.61× to 2.96×</h3>
          </div>
          <div className="progress-key" aria-label="Chart series">
            <span><i className="progress-swatch progress-kda" /> KDA best</span>
            <span><i className="progress-swatch progress-cake" /> CAKE</span>
            <span><i className="progress-swatch progress-int21" /> INT21 reference</span>
          </div>
        </div>
        <div className="progress-scroll">
          <svg viewBox="0 0 1090 392" role="img" aria-label="KDA best rises from 1.61 times on July 21 to 2.96 times on September 12. Dashed arrows mark Humanize2 at 2.45 times on August 14; TIRx at 2.54 times on August 30, 2.93 times on September 2, and 2.96 times on September 12; and CAKE plus KDA at 2.56 times on September 1 and CAKE at 2.94 times on September 6. INT21 is 1.5 times on June 16.">
            <defs>
              <marker id="progress-humanize-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#c7ff3d" /></marker>
              <marker id="progress-tirx-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#a9b1ad" /></marker>
              <marker id="progress-cake-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#ffbb33" /></marker>
            </defs>
            {milestones.map((milestone) => (
              <g key={milestone.label} className={`progress-milestone progress-milestone-${milestone.tone}`}>
                <text className="progress-milestone-label" x={milestone.labelX} y="24" textAnchor="middle">{milestone.label}</text>
                <text className="progress-milestone-detail" x={milestone.labelX} y="42" textAnchor="middle">{milestone.detail}</text>
                {milestone.arrows.map((arrow) => (
                  <path
                    key={arrow.index}
                    className="progress-milestone-arrow"
                    d={`M${arrow.startX} 57 C${arrow.startX} 91, ${x(arrow.index)} 85, ${x(arrow.index)} ${y(arrow.score) - 16}`}
                    markerEnd={`url(#progress-${milestone.tone}-arrow)`}
                  />
                ))}
              </g>
            ))}
            {[1.5, 2, 2.5, 3].map((tick) => (
              <g key={tick}>
                <line className="progress-gridline" x1={left} x2={right} y1={y(tick)} y2={y(tick)} />
                <text className="progress-axis" x={left - 12} y={y(tick) + 4} textAnchor="end">{tick.toFixed(1)}×</text>
              </g>
            ))}
            <path className="progress-line progress-line-cake" d={line('cake')} />
            <path className="progress-line progress-line-kda" d={line('kda')} />
            {milestones.flatMap((milestone) => milestone.arrows.map((arrow) => (
              <circle key={`${milestone.label}-${arrow.index}`} className={`progress-milestone-ring progress-milestone-${milestone.tone}`} cx={x(arrow.index)} cy={y(arrow.score)} r="11" />
            )))}
            {history.map((row, index) => (
              <g key={row.date}>
                <text className="progress-axis" x={x(index)} y="378" textAnchor="middle">{row.date}</text>
                {(['cake', 'kda'] as const).map((method) => {
                  const point = value(row, method);
                  return point === undefined ? null : (
                    <circle key={method} className={`progress-point progress-point-${method}`} cx={x(index)} cy={y(point)} r="5">
                      <title>{`${row.date}: ${method === 'kda' ? 'KDA best' : 'CAKE'} ${point}×`}</title>
                    </circle>
                  );
                })}
                {value(row, 'int21') !== undefined && (
                  <circle className="progress-point progress-point-int21" cx={x(index)} cy={y(value(row, 'int21')!)} r="5">
                    <title>6/16: INT21 1.5×</title>
                  </circle>
                )}
              </g>
            ))}
            <text className="progress-end-label" x={right - 8} y={y(2.96) + 37} textAnchor="end">2.96×</text>
          </svg>
        </div>
      </div>
      <figcaption className="progress-caption">
        <div className="progress-captions">
          <div className="progress-caption-cake">
            <span>FINAL RESULT · CAKE-PTX</span>
            <strong>KDA + CAKE <b>2.94×</b></strong>
            <small>Agent-guided CAKE IR tuning, compiled to PTX.</small>
          </div>
          <div className="progress-caption-tirx">
            <span>FINAL RESULT · TIRx</span>
            <strong>KDA + TIRx <b>2.96×</b></strong>
            <small>The strongest validated result on B300.</small>
          </div>
        </div>
        <p>Dashed arrows mark the dated Humanize2, TIRx, and CAKE contributions. The green line tracks KDA best; the orange line tracks CAKE separately. The captions identify the final method results reported in the accompanying PDF.</p>
      </figcaption>
    </figure>
  );
}
