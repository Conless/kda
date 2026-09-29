import { internalPath } from '../internal-path';
import { BLOG_DESCRIPTION, feedUrl, formatPostDate, postPath, posts } from './posts';

export default function BlogIndex() {
  const [latest] = posts;
  const years = [...new Set(posts.map((post) => post.date.slice(0, 4)))];

  return (
    <>
      <header className="blog-hero shell">
        <div className="eyebrow"><span /> KDA Blog</div>
        <h1 className="blog-title">Notes from the<br /><em>kernel design loop</em></h1>
        <div className="blog-hero-foot">
          <p className="blog-lede">{BLOG_DESCRIPTION}</p>
          <div className="blog-hero-links">
            <a className="text-link" href={feedUrl}>Subscribe via RSS <span aria-hidden="true">↗</span></a>
            <a className="text-link" href={internalPath('/#submit')}>Request a kernel <span aria-hidden="true">↗</span></a>
          </div>
        </div>
      </header>

      {latest && (
        <section className="shell blog-featured" aria-label="Latest post">
          <a className="featured-card" href={internalPath(postPath(latest.slug))}>
            <div className="featured-copy">
              <p className="section-label">LATEST POST</p>
              <p className="post-meta">
                <time dateTime={latest.date}>{formatPostDate(latest.date)}</time> · {latest.readingTime}
              </p>
              <h2>{latest.title}</h2>
              <p className="featured-description">{latest.description}</p>
              <ul className="tag-list" aria-label="Tags">
                {latest.tags.map((tag) => <li key={tag}>{tag}</li>)}
              </ul>
              <span className="featured-cta">Read the post <span aria-hidden="true">→</span></span>
            </div>
            <div className="featured-visual" aria-hidden="true">
              <div className="pipeline-topline">
                <span>HIGHLIGHT</span>
                <span className="live"><i /> NEW</span>
              </div>
              <strong>{latest.highlight.value}</strong>
              <span>{latest.highlight.label}</span>
              <div className="featured-bars">
                {[0.93, 0.92, 0.89, 0.31].map((width, index) => (
                  <i key={index} style={{ '--w': width, '--i': index } as React.CSSProperties} />
                ))}
              </div>
            </div>
          </a>
        </section>
      )}

      <section className="shell blog-archive" aria-labelledby="archive-heading">
        <div className="blog-archive-head">
          <p className="section-label" id="archive-heading">ALL POSTS</p>
          <span>{posts.length} {posts.length === 1 ? 'post' : 'posts'}</span>
        </div>
        {years.map((year) => (
          <div className="archive-year" key={year}>
            <h2>{year}</h2>
            <ol>
              {posts.filter((post) => post.date.startsWith(year)).map((post) => (
                <li key={post.slug}>
                  <a className="archive-row" href={internalPath(postPath(post.slug))}>
                    <time dateTime={post.date}>{formatPostDate(post.date, 'short').replace(`, ${year}`, '')}</time>
                    <div>
                      <h3>{post.title}</h3>
                      <p>{post.description}</p>
                    </div>
                    <span className="archive-tags">{post.tags.join(' · ')}</span>
                    <span className="archive-arrow" aria-hidden="true">→</span>
                  </a>
                </li>
              ))}
            </ol>
          </div>
        ))}
      </section>
    </>
  );
}
