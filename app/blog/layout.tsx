import type { Metadata } from 'next';
import Link from 'next/link';
import { BLOG_DESCRIPTION, BLOG_TITLE, blogUrl, feedUrl, siteUrl } from './posts';
import './blog.css';

const REPOSITORY_URL = 'https://github.com/NVlabs/kda';
const HUMANIZE_URL = 'https://github.com/humanfia/humanize2';

export const metadata: Metadata = {
  title: {
    default: `${BLOG_TITLE} — Kernel Design Agents`,
    template: `%s | ${BLOG_TITLE}`,
  },
  description: BLOG_DESCRIPTION,
  alternates: {
    canonical: blogUrl,
    types: { 'application/rss+xml': [{ url: feedUrl, title: BLOG_TITLE }] },
  },
  openGraph: {
    type: 'website',
    url: blogUrl,
    title: `${BLOG_TITLE} — Kernel Design Agents`,
    description: BLOG_DESCRIPTION,
    images: [{ url: `${siteUrl}/og.png`, width: 1200, height: 630, alt: 'Kernel Design Agents' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: `${BLOG_TITLE} — Kernel Design Agents`,
    description: BLOG_DESCRIPTION,
    images: [`${siteUrl}/og.png`],
  },
};

export default function BlogLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <>
      <nav className="nav shell" aria-label="Primary navigation">
        <Link className="brand" href="/" aria-label="Kernel Design Agents home">
          <span className="brand-mark" aria-hidden="true">KDA</span>
          <span>Kernel Design <b>Agents</b></span>
        </Link>
        <div className="nav-links">
          <Link href="/#process">How it works</Link>
          <Link href="/#achievements">Achievements</Link>
          <Link href="/#faq">FAQ</Link>
          <Link href="/blog/">Blog</Link>
          <a className="nav-cta" href={REPOSITORY_URL}>View GitHub <span aria-hidden="true">↗</span></a>
        </div>
      </nav>

      <main className="blog">{children}</main>

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
    </>
  );
}
