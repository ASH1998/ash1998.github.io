# Ashutosh Mishra — Portfolio

Live at [ash1998.github.io](https://ash1998.github.io/).

The homepage is a responsive portfolio with a grainy midnight background and ten interactive particle forms. Desktop scrolling uses fixed forms 1–5: Overview, Experience, Practice, Skills, and Achievements onward. Shapes never cycle automatically. Click the number/name label to explore extra forms 6–10; left/right arrow keys also browse those extras. Clicking the sculpture triggers its own effect (scatter, ripple, twist, and more) without selecting another shape. Hover illuminates surfaces and dragging rotates the current form. Motion can be paused and respects reduced-motion preferences.

## Editing the portfolio

- `index.html`: portfolio content, navigation, metadata, and contact links.
- `static/portfolio/styles.css`: visual design and responsive layout.
- `static/portfolio/app.js`: particle geometry, lighting, motion, and interactions.
- `static/portfolio/renderer.js`: GPU point rendering with a Canvas fallback.
- `blog/index.html` and `_layouts/journal.html`: blog archive and shared journal layout.
- `_layouts/post.html`: article reading layout; original Markdown remains in `_posts`.
- `static/portfolio/journal.css` and `journal.js`: blog typography, responsive layout, and texture.

The homepage uses HTML, CSS, JavaScript, and a Liquid loop that shows the four newest published posts from the same `site.posts` collection as the blog archive. GitHub Pages renders this automatically when publishing `master` at its root using the existing Jekyll setup; a full local preview needs Jekyll to render the recent-post list. No npm installation is needed. The blog archive automatically lists the original posts with matching dark article layouts. Post URLs, Markdown content, comments, and sharing remain available.

[Blog](https://ash1998.github.io/blog/) · [GitHub](https://github.com/ASH1998)

## Publishing a blog post

Create a Markdown file under `_posts` named `YYYY-MM-DD-your-post-slug.md`:

```yaml
---
layout: post
title: "Your article title"
date: 2026-09-19 12:00:00 +0530
desc: "A short, useful summary that also helps readers find this post."
categories: [machine-learning]
tags: [python, transformers]
---
```

Write the article below that front matter, commit, and push to `master`. GitHub Pages rebuilds the article, archive, and search index automatically. Dates in the future are not published until a build after that date. Keep unfinished posts in `_drafts`.

The archive generates static pages of 12 entries using `jekyll-paginate`; browsing works without JavaScript. Search loads `/blog/search.json` only after a query, matches titles, summaries (up to 360 characters), categories, and tags, and renders at most 12 results at a time. Every query term must match; title matches rank first and ties retain newest-first order. Search URLs can be bookmarked or shared. Full article bodies are not downloaded or searched. New posts need no manual index maintenance.

This metadata index is suitable for thousands of text posts; its download and search work grow with the number of entries. Compress article media and monitor build time/site size as the archive grows. GitHub Pages currently caps the published site at 1 GB and builds at 10 minutes: https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits

Run search regression checks with `node --test tests/blog-search.test.cjs` (including a 5,000-entry fixture).
