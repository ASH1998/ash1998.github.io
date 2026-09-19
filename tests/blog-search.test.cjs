const { test } = require('node:test');
const assert = require('node:assert/strict');
const { prepare, search, paginate } = require('../static/portfolio/blog-search.js');

test('all query terms must match across metadata fields; titles rank above summaries', () => {
  const posts = prepare([
    { title: 'Training notes', description: 'Attention for translation', categories: ['ML'], tags: ['pytorch'] },
    { title: 'Attention models', description: 'Translation', categories: ['ML'], tags: ['pytorch'] },
    { title: 'Attention models', description: 'Vision', categories: [], tags: [] }
  ]);
  assert.deepEqual(search(posts, 'ATTENTION pytorch').map(post => post.title), ['Attention models', 'Training notes']);
  assert.equal(search(posts, '  ').length, 3);
  assert.equal(search(posts, 'no-such-topic').length, 0);
});

test('accent insensitive literal search handles punctuation and keeps stable ties', () => {
  const posts = prepare([{ title: 'Café C++ [notes]', description: '' }, { title: 'Cafe C++ [notes]', description: '' }]);
  assert.equal(search(posts, 'cafe C++ [notes]').length, 2);
  assert.equal(search(posts, 'cafe')[0], posts[0]);
  assert.equal(search(posts, '.*').length, 0);
});

test('5,000 entries remain paginated with no missing or duplicate results', () => {
  const posts = prepare(Array.from({ length: 5000 }, (_, i) => ({ title: `Research ${i}`, description: 'Machine learning notes', categories: ['AI'], tags: ['python'] })));
  const start = performance.now();
  const matches = search(posts, 'research python');
  assert.equal(matches.length, 5000);
  assert.equal(paginate(matches, 1).pages, 417);
  assert.equal(paginate(matches, 1).items.length, 12);
  assert.equal(paginate(matches, 417).items.length, 8);
  const seen = new Set();
  for (let page = 1; page <= 417; page++) {
    for (const post of paginate(matches, page).items) { assert.ok(!seen.has(post.title)); seen.add(post.title); }
  }
  assert.equal(seen.size, 5000);
  console.log(`5,000-entry search and page traversal: ${(performance.now() - start).toFixed(1)} ms`);
});

test('page boundaries tolerate invalid and out-of-range URL values', () => {
  const posts = prepare(Array.from({ length: 25 }, (_, i) => ({ title: String(i) })));
  assert.equal(paginate(posts, -20).page, 1);
  assert.equal(paginate(posts, 'oops').page, 1);
  assert.equal(paginate(posts, 2.9).page, 2);
  assert.equal(paginate(posts, 9999).page, 3);
  assert.deepEqual(paginate([], 9), { items: [], page: 1, pages: 1, start: 0 });
});
