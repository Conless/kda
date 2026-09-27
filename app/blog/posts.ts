import type { Metadata } from 'next';

export const siteUrl = process.env.SITE_URL ?? 'https://nvlabs.github.io/kda';
export const BLOG_TITLE = 'KDA Blog';
export const BLOG_DESCRIPTION =
  'Results, failure modes, and lessons from building agents that write, verify, and tune GPU kernels.';

export type BlogPost = {
  slug: string;
  title: string;
  description: string;
  /** ISO date, YYYY-MM-DD. */
  date: string;
  authors: string;
  readingTime: string;
  tags: readonly string[];
  highlight: { value: string; label: string };
};

// Newest first. To publish a post, add app/blog/<slug>/page.tsx and register it here.
export const posts: readonly BlogPost[] = [
  {
    slug: '2026-09-27-kda-for-kda',
    title: 'KDA²: Kernel Design Agents Optimize Kimi Delta Attention',
    description:
      'Our agents wrote Kimi Delta Attention kernels that run up to 2.96× faster than FlashKDA on B300 with a tenth of its state error. Here is how, and how the agents tried to cheat along the way.',
    date: '2026-09-27',
    authors: 'Kernel Design Agents team · NVIDIA',
    readingTime: '12 min read',
    tags: ['Results', 'Kimi Delta Attention', 'Reward hacking'],
    highlight: { value: '2.96×', label: 'geomean speedup over FlashKDA on B300' },
  },
];

export const blogUrl = `${siteUrl}/blog/`;
export const feedUrl = `${siteUrl}/blog/rss.xml`;

export function postPath(slug: string) {
  return `/blog/${slug}/`;
}

export function getPost(slug: string) {
  const post = posts.find((candidate) => candidate.slug === slug);
  if (!post) throw new Error(`Unknown blog post: ${slug}`);
  return post;
}

export function formatPostDate(date: string, month: 'long' | 'short' = 'long') {
  return new Date(`${date}T00:00:00Z`).toLocaleDateString('en-US', {
    timeZone: 'UTC',
    year: 'numeric',
    month,
    day: 'numeric',
  });
}

export function postMetadata(post: BlogPost): Metadata {
  const url = `${siteUrl}${postPath(post.slug)}`;
  return {
    title: post.title,
    description: post.description,
    authors: [{ name: post.authors }],
    alternates: {
      canonical: url,
      types: { 'application/rss+xml': [{ url: feedUrl, title: BLOG_TITLE }] },
    },
    openGraph: {
      type: 'article',
      url,
      title: post.title,
      description: post.description,
      publishedTime: post.date,
      tags: [...post.tags],
      images: [{ url: `${siteUrl}/og.png`, width: 1200, height: 630, alt: 'Kernel Design Agents' }],
    },
    twitter: {
      card: 'summary_large_image',
      title: post.title,
      description: post.description,
      images: [`${siteUrl}/og.png`],
    },
  };
}
