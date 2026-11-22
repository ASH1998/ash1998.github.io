const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync(require.resolve('../static/portfolio/app.js'), 'utf8');
function scene({ mobile = true } = {}) {
  let now = 0, nextId = 0, points;
  const frames = new Map(), nodes = new Map(), timers = new Map(), observers = [];
  function node(selector) {
    if (!nodes.has(selector)) nodes.set(selector, {
      style: {}, dataset: {}, handlers: {}, textContent: '',
      after() {}, setAttribute() {},
      addEventListener(type, fn) { this.handlers[type] = fn; },
      getBoundingClientRect() { return { left: 0, top: 0, width: 400, height: 400 }; },
      setPointerCapture() {}, hasPointerCapture() { return false; }
    });
    return nodes.get(selector);
  }
  vm.runInNewContext(source, {
    Math, Float32Array, performance: { now: () => now }, devicePixelRatio: 1,
    document: {
      hidden: false, querySelector: node, querySelectorAll: () => [], addEventListener() {},
      createElement: () => ({ getContext: () => ({ createImageData: () => ({ data: new Uint8Array(180 * 180 * 4) }), putImageData() {} }), toDataURL: () => '' })
    },
    window: { createParticleRenderer: () => ({ resize() {}, draw(data) { points = data; } }) },
    matchMedia: query => ({ matches: mobile && query.includes('max-width'), addEventListener() {} }),
    IntersectionObserver: class { constructor(callback, options) { observers.push({ callback, options }); } observe() {} }, ResizeObserver: class { observe() {} },
    requestAnimationFrame(fn) { frames.set(++nextId, fn); return nextId; },
    cancelAnimationFrame(id) { frames.delete(id); },
    setTimeout(fn) { timers.set(++nextId, fn); return nextId; }, clearTimeout(id) { timers.delete(id); }
  });
  return {
    node, frames,
    visit(id) {
      observers.find(o => o.options.rootMargin === '-32% 0px -43% 0px').callback([{ isIntersecting: true, target: { id } }]);
      for (const fn of timers.values()) fn();
      timers.clear();
    },
    advance(seconds, hz = 60) {
      for (let i = 0; i < Math.round(seconds * hz); i++) {
        now += 1000 / hz;
        const callbacks = [...frames.values()]; frames.clear();
        for (const fn of callbacks) fn(now);
      }
    },
    next() { node('#shape-name').handlers.click(); },
    click() { node('#sculpture').handlers.click({ detail: 0 }); },
    pause() { node('#motion').handlers.click(); },
    snapshot() { return Float32Array.from(points); }
  };
}

test('scrolling keeps the sphere on every section and never overrides a chosen form', () => {
  const s = scene({ mobile: false });
  const sections = ['overview','experience','practice','skills','achievements','writing','contact'];
  for (const section of [...sections, ...sections.toReversed()]) {
    s.visit(section); assert.equal(s.node('#particles').dataset.shape, 'Sphere');
  }
  s.visit('skills'); s.next();
  assert.equal(s.node('#particles').dataset.shape, 'Saturn');
  for (const section of sections) {
    s.visit(section); assert.equal(s.node('#particles').dataset.shape, 'Saturn', 'scrolling keeps the form the visitor chose');
  }
});

test('idle time never changes either a scroll form or an extra form', () => {
  const s = scene();
  s.advance(25, 30); assert.equal(s.node('#particles').dataset.shape, 'Sphere');
  s.next(); s.click(); s.advance(25, 30); assert.equal(s.node('#particles').dataset.shape, 'Saturn');
  assert.equal(s.node('#particles').dataset.effect, 'idle');
});

test('number label and keyboard activation cycle through every form in order and wrap', () => {
  const s = scene(), expected = ['Saturn','Atom','Wormhole','Lorenz','Network','Tesseract','Gyroid','Sphere','Saturn'];
  for (const shape of expected) { s.next(); assert.equal(s.node('#particles').dataset.shape, shape); }
  s.node('#sculpture').handlers.keydown({key:'ArrowLeft',preventDefault(){}});
  assert.equal(s.node('#particles').dataset.shape, 'Sphere');
  s.next(); assert.equal(s.node('#particles').dataset.shape, 'Saturn');
});

test('mobile section navigation leaves the hero sculpture unchanged', () => {
  const s = scene(); s.visit('overview'); s.visit('skills'); s.visit('achievements');
  assert.equal(s.node('#particles').dataset.shape, 'Sphere');
});

test('drag inertia follows elapsed time across 30, 60 and 120 Hz displays', () => {
  function run(hz) {
    const s = scene(), handlers = s.node('#sculpture').handlers;
    handlers.pointerdown({ button: 0, clientX: 200, clientY: 200, pointerType: 'mouse', pointerId: 1 });
    handlers.pointermove({ clientX: 260, clientY: 230 });
    s.advance(.1, 60);
    handlers.pointerup({ pointerId: 1 });
    s.advance(2, hz);
    return s.snapshot();
  }
  const reference = run(60);
  for (const hz of [30, 120]) {
    const actual = run(hz);
    let maxError = 0;
    for (let i = 0; i < actual.length; i += 5) {
      maxError = Math.max(maxError, Math.hypot(actual[i] - reference[i], actual[i + 1] - reference[i + 1]));
    }
    assert.ok(maxError < .1, `${hz} Hz position error: ${maxError}px`);
  }
});

test('every form\'s click effect works without selecting another shape', () => {
  const s = scene({ mobile: false });
  const effects = ['scatter','accrete','cloud','warp','butterfly','pass','hyper','split'];
  for (let i = 0; i < 16; i++) {
    s.advance(i % 2 ? 3 : .3);
    const shape = s.node('#particles').dataset.shape;
    s.click();
    assert.equal(s.node('#particles').dataset.shape, shape);
    assert.equal(s.node('#particles').dataset.effect, effects[i % 8]);
    s.advance(.5);
    assert.ok(s.snapshot().every(Number.isFinite));
    s.next();
  }
});

test('pause stops frame scheduling, permits shape selection, and resumes cleanly', () => {
  const s = scene(); s.advance(.5); s.pause();
  const paused = s.snapshot(); s.advance(1);
  assert.deepEqual(s.snapshot(), paused); assert.equal(s.frames.size, 0);
  s.click(); assert.deepEqual(s.snapshot(), paused, 'sculpture effects respect pause');
  s.next(); assert.equal(s.node('#particles').dataset.shape, 'Saturn');
  assert.ok(s.snapshot().every(Number.isFinite));
  s.pause(); s.advance(.2); assert.equal(s.frames.size, 1);
  assert.notDeepEqual(s.snapshot(), paused);
});

test('holding a drag still before release does not launch stale pointer velocity', () => {
  const s = scene(), handlers = s.node('#sculpture').handlers;
  handlers.pointerdown({ button: 0, clientX: 200, clientY: 200, pointerType: 'mouse', pointerId: 1 });
  handlers.pointermove({ clientX: 300, clientY: 240 });
  s.advance(1);
  handlers.pointerup({ pointerId: 1 });
  const before = s.snapshot(); s.advance(1 / 60); const after = s.snapshot();
  let movement = 0;
  for (let i = 0; i < after.length; i += 5) movement = Math.max(movement, Math.hypot(after[i] - before[i], after[i + 1] - before[i + 1]));
  assert.ok(movement < .3, `release moved particles by ${movement}px`);
  s.click(); assert.equal(s.node('#particles').dataset.shape, 'Sphere');
  assert.equal(s.node('#particles').dataset.effect, undefined, 'a drag release does not trigger the click effect');
});

test('a sculpture click visibly changes the current form while a label click selects another', () => {
  const baseline = scene(), interactive = scene();
  baseline.advance(1); interactive.click(); interactive.advance(1);
  assert.equal(interactive.node('#particles').dataset.shape, 'Sphere');
  const before = baseline.snapshot(), after = interactive.snapshot();
  let movement = 0;
  for (let i = 0; i < after.length; i += 5) movement = Math.max(movement, Math.hypot(after[i] - before[i], after[i + 1] - before[i + 1]));
  assert.ok(movement > 5, 'scatter visibly displaces particles');
  interactive.next();
  assert.equal(interactive.node('#particles').dataset.shape, 'Saturn');
  assert.equal(interactive.node('#particles').dataset.effect, 'idle', 'changing forms clears the old effect');
});
