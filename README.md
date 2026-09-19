# Ashutosh Mishra — Portfolio

Live at [ash1998.github.io](https://ash1998.github.io/).

The homepage is a responsive portfolio with a grainy midnight background and ten interactive particle forms. Hover to illuminate surfaces, drag to rotate, and click for each form's effect. Use the shape label or left/right arrow keys to change forms. Motion can be paused and respects reduced-motion preferences.

## Editing the portfolio

- `index.html`: portfolio content, navigation, metadata, and contact links.
- `static/portfolio/styles.css`: visual design and responsive layout.
- `static/portfolio/app.js`: particle geometry, lighting, motion, and interactions.
- `static/portfolio/renderer.js`: GPU point rendering with a Canvas fallback.
- `blog/blog.html` and `_layouts/journal.html`: blog archive and shared journal layout.
- `_layouts/post.html`: article reading layout; original Markdown remains in `_posts`.
- `static/portfolio/journal.css` and `journal.js`: blog typography, responsive layout, and texture.

The homepage uses plain HTML, CSS, and JavaScript; no compilation or npm installation is needed for homepage changes. GitHub Pages publishes the repository from `master` at its root using the existing Jekyll setup. The blog archive automatically lists the original posts with matching dark article layouts. Post URLs, Markdown content, comments, and sharing remain available.

[Blog](https://ash1998.github.io/blog/) · [GitHub](https://github.com/ASH1998)
