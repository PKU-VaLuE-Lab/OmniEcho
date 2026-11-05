/* Decorative, locally rendered spatial field. No external graphics dependency. */
(() => {
  'use strict';

  const starCanvas = document.getElementById('starfield');
  const orbCanvas = document.getElementById('spatial-orb');
  const starContext = starCanvas?.getContext('2d');
  const orbContext = orbCanvas?.getContext('2d');
  if (!starContext && !orbContext) return;

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const finePointer = window.matchMedia('(pointer: fine)');
  const motionButton = document.querySelector('.motion-control');
  const hero = document.querySelector('.hero');
  let paused = reduced.matches;
  let time = 0;
  let frame = null;
  let previous = 0;
  let lastDraw = 0;
  let width = 0;
  let height = 0;
  let scale = 1;
  let starWidth = 0;
  let starHeight = 0;
  let starVisible = true;
  let orbVisible = true;
  const pointer = { x: 0, y: 0, targetX: 0, targetY: 0 };

  let seed = 1701;
  function random() {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  }
  function gaussian() {
    return Math.sqrt(-2 * Math.log(Math.max(random(), .0001))) * Math.cos(Math.PI * 2 * random());
  }

  const stars = [];
  for (let i = 0; i < 470; i++) {
    const band = i > 245;
    const x = random();
    const y = band ? .15 + (1 - x) * .44 + gaussian() * .074 : random();
    stars.push({ x, y, size: i % 19 === 0 ? 1.45 : .35 + random() * .6,
      phase: random() * Math.PI * 2, opacity: .15 + random() * .62,
      depth: .25 + random() * .75, warm: i % 4 === 0 });
  }

  const spherePoints = [];
  for (let i = 0; i < 1350; i++) {
    const y = 1 - 2 * i / 1349;
    const r = Math.sqrt(1 - y * y);
    const angle = i * Math.PI * (3 - Math.sqrt(5));
    spherePoints.push([188 * r * Math.cos(angle), 188 * y, 188 * r * Math.sin(angle)]);
  }
  const latitudes = [];
  for (let latitude = -75; latitude <= 75; latitude += 15) {
    const phi = latitude * Math.PI / 180;
    const line = [];
    for (let j = 0; j <= 120; j++) {
      const angle = j * Math.PI / 60;
      line.push([188 * Math.cos(phi) * Math.cos(angle), 188 * Math.sin(phi), 188 * Math.cos(phi) * Math.sin(angle)]);
    }
    latitudes.push(line);
  }
  const meridians = [];
  for (let i = 0; i < 12; i++) {
    const angle = i * Math.PI / 12;
    const line = [];
    for (let j = 0; j <= 120; j++) {
      const phi = j * Math.PI / 60;
      line.push([188 * Math.sin(phi) * Math.cos(angle), 188 * Math.cos(phi), 188 * Math.sin(phi) * Math.sin(angle)]);
    }
    meridians.push(line);
  }
  const orbitPoints = [[], [], []];
  for (let j = 0; j <= 180; j++) {
    const a = j * Math.PI / 90;
    orbitPoints[0].push([Math.cos(a) * 283, Math.sin(a) * -75, Math.sin(a) * 235]);
    orbitPoints[1].push([Math.cos(a) * 234, Math.sin(a) * 153, Math.sin(a) * 185]);
    orbitPoints[2].push([Math.cos(a) * 251, Math.sin(a) * -185, Math.sin(a) * 100]);
  }

  function setupCanvas(canvas, ctx) {
    const ratio = Math.min(devicePixelRatio || 1, 2);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = Math.round(w * ratio);
    canvas.height = Math.round(h * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return [w, h];
  }
  function resize() {
    if (starContext) [starWidth, starHeight] = setupCanvas(starCanvas, starContext);
    if (orbContext) {
      [width, height] = setupCanvas(orbCanvas, orbContext);
      scale = Math.min(width / 620, height / 590);
    }
    draw();
  }

  function starGlow(ctx, x, y, radius, rgb, opacity) {
    const g = ctx.createRadialGradient(x, y, 0, x, y, radius);
    g.addColorStop(0, `rgba(${rgb},${opacity})`);
    g.addColorStop(.18, `rgba(${rgb},${opacity * .4})`);
    g.addColorStop(1, `rgba(${rgb},0)`);
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  function drawStars() {
    if (!starContext) return;
    const ctx = starContext;
    ctx.clearRect(0, 0, starWidth, starHeight);
    const limit = starWidth < 600 ? 210 : stars.length;
    for (let i = 0; i < limit; i++) {
      const star = stars[i];
      const x = star.x * starWidth + pointer.x * star.depth * 7;
      const y = star.y * starHeight + pointer.y * star.depth * 5;
      const opacity = star.opacity * (.77 + .23 * Math.sin(star.phase + time * .38));
      const rgb = star.warm ? '213,199,255' : '205,235,255';
      if (star.size > 1.2) {
        starGlow(ctx, x, y, 9, rgb, opacity * .28);
        ctx.strokeStyle = `rgba(${rgb},${opacity * .5})`;
        ctx.lineWidth = .6;
        ctx.beginPath();ctx.moveTo(x - 4, y);ctx.lineTo(x + 4, y);
        ctx.moveTo(x, y - 4);ctx.lineTo(x, y + 4);ctx.stroke();
      }
      ctx.fillStyle = `rgba(${rgb},${opacity})`;
      ctx.beginPath();ctx.arc(x, y, star.size, 0, Math.PI * 2);ctx.fill();
    }
  }

  let sinA = 0, cosA = 1, sinB = 0, cosB = 1;
  function project(point) {
    const [x, y, z] = point;
    const x1 = x * cosA + z * sinA;
    const z1 = -x * sinA + z * cosA;
    const y1 = y * cosB - z1 * sinB;
    const z2 = y * sinB + z1 * cosB;
    const perspective = 890 / (890 + z2);
    return { x: width * .51 + x1 * scale * perspective + pointer.x * 7,
      y: height * .49 + y1 * scale * perspective + pointer.y * 5, z: z2 };
  }
  function line(points, rgb, opacity, thickness, frontOnly = null) {
    const ctx = orbContext;
    ctx.lineWidth = thickness * Math.max(scale, .75);
    for (let i = 1; i < points.length; i++) {
      const p = points[i - 1], q = points[i];
      const depth = (p.z + q.z) / 2;
      if (frontOnly === true && depth > 0 || frontOnly === false && depth <= 0) continue;
      const intensity = Math.max(.18, .63 - depth / 440);
      ctx.strokeStyle = `rgba(${rgb},${opacity * intensity})`;
      ctx.beginPath();ctx.moveTo(p.x, p.y);ctx.lineTo(q.x, q.y);ctx.stroke();
    }
  }
  function drawOrb() {
    if (!orbContext || !width || !height) return;
    const ctx = orbContext;
    ctx.clearRect(0, 0, width, height);
    const a = .45 + time * .028 + pointer.x * .14;
    const b = -.23 + pointer.y * .10;
    sinA = Math.sin(a);cosA = Math.cos(a);sinB = Math.sin(b);cosB = Math.cos(b);
    const center = project([0, 0, 0]);
    const radius = 192 * scale;

    // Large soft atmospheric light is separate from the sharp sphere surface.
    starGlow(ctx, center.x, center.y, radius * 1.5, '154,168,218', .08);
    starGlow(ctx, center.x - radius * .4, center.y + radius * .25, radius * 1.1, '151,189,206', .08);
    const orbits = orbitPoints.map(points => points.map(project));
    orbits.forEach((points, i) => line(points, i === 0 ? '132,119,176' : '121,145,181', .44, 1, false));

    const sphere = ctx.createRadialGradient(center.x - radius * .34, center.y - radius * .32, 0, center.x, center.y, radius);
    sphere.addColorStop(0, 'rgba(188,209,225,.15)');
    sphere.addColorStop(.48, 'rgba(193,213,228,.08)');
    sphere.addColorStop(.84, 'rgba(178,196,220,.12)');
    sphere.addColorStop(.98, 'rgba(190,197,227,.08)');
    sphere.addColorStop(1, 'rgba(190,210,235,0)');
    ctx.fillStyle = sphere;ctx.beginPath();ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);ctx.fill();

    const rim = ctx.createLinearGradient(center.x - radius, center.y, center.x + radius, center.y);
    rim.addColorStop(0, '#8197bb65');rim.addColorStop(.45, '#a0b2cf25');rim.addColorStop(1, '#9a8aba70');
    ctx.strokeStyle = rim;ctx.lineWidth = 1.3 * scale;
    ctx.shadowColor = '#d6dcef';ctx.shadowBlur = 13 * scale;
    ctx.beginPath();ctx.arc(center.x, center.y, radius * .992, 0, Math.PI * 2);ctx.stroke();ctx.shadowBlur = 0;

    for (const points of latitudes) line(points.map(project), '102,136,166', .28, .65);
    for (const points of meridians) line(points.map(project), '110,133,180', .25, .6);

    // The golden-angle point distribution creates depth without a bitmap texture.
    for (let i = 0; i < spherePoints.length; i++) {
      const p = project(spherePoints[i]);
      const front = (1 - p.z / 205) / 2;
      const alpha = Math.min(.94, .10 + front * .69);
      ctx.fillStyle = `rgba(${p.x < center.x ? '105,142,172' : '140,127,179'},${alpha})`;
      ctx.beginPath();ctx.arc(p.x, p.y, (.35 + front * .67) * scale, 0, Math.PI * 2);ctx.fill();
    }

    orbits.forEach((points, i) => line(points, i === 0 ? '137,117,179' : '103,137,175', i === 0 ? .7 : .46, i === 0 ? 1.1 : .8, true));
    // A short moving trail makes the primary orbit read as a spatial signal.
    const phase = (time * .065 + .59) % (Math.PI * 2);
    const trail = [];
    for (let i = 0; i < 19; i++) {
      const t = phase - i * .018;
      trail.push(project([Math.cos(t) * 283, Math.sin(t) * -75, Math.sin(t) * 235]));
    }
    line(trail, '118,107,170', .65, 1.6);

    const source = project([145, -151, -108]);
    const vision = project([-146, 117, -82]);
    line([center, source], '108,146,178', .38, 1);
    ctx.setLineDash([3 * scale, 5 * scale]);
    line([center, vision], '139,123,175', .36, 1);ctx.setLineDash([]);
    [source, vision, center, trail[0]].forEach((point, i) => {
      const rgb = i === 1 || i === 3 ? '161,147,191' : '134,165,194';
      starGlow(ctx, point.x, point.y, (i === 2 ? 49 : 25) * scale, rgb, i === 2 ? .27 : .18);
      ctx.fillStyle = i === 1 || i === 3 ? '#8a78ac' : '#648bad';ctx.beginPath();ctx.arc(point.x, point.y, (i === 2 ? 4.3 : 2.8) * scale, 0, Math.PI * 2);ctx.fill();
    });

    for (let i = 0; i < 3; i++) {
      const wave = ((time * .10 + i / 3) % 1);
      ctx.lineWidth = .8 * scale;
      ctx.strokeStyle = `rgba(107,143,180,${(1 - wave) * .23})`;
      ctx.beginPath();ctx.arc(center.x, center.y, (13 + wave * 56) * scale, 0, Math.PI * 2);ctx.stroke();
    }
    // A distant tilted latitude gives the entire composition a horizon.
    ctx.lineWidth = .65 * scale;ctx.strokeStyle = 'rgba(148,166,195,.17)';
    ctx.beginPath();ctx.ellipse(center.x, center.y + radius * 1.07, radius * 1.35, radius * .22, -.10, Math.PI, Math.PI * 2);ctx.stroke();
  }

  function draw() {
    if (starVisible) drawStars();
    if (orbVisible) drawOrb();
  }
  function tick(now) {
    frame = null;
    if (paused || document.hidden || (!starVisible && !orbVisible)) { previous = 0; return; }
    if (previous) time += Math.min((now - previous) / 1000, .05);
    previous = now;
    if (now - lastDraw >= 32) {
      pointer.x += (pointer.targetX - pointer.x) * .075;
      pointer.y += (pointer.targetY - pointer.y) * .075;
      draw();lastDraw = now;
    }
    frame = requestAnimationFrame(tick);
  }
  function play() {
    if (!paused && !document.hidden && (starVisible || orbVisible) && frame === null) frame = requestAnimationFrame(tick);
  }
  function setPaused(value) {
    paused = value;
    document.documentElement.classList.toggle('motion-paused', paused);
    document.documentElement.dataset.motion = paused ? 'paused' : 'running';
    if (motionButton) {
      motionButton.hidden = false;
      motionButton.setAttribute('aria-pressed', String(paused));
      motionButton.setAttribute('aria-label', paused ? 'Resume decorative animation' : 'Pause decorative animation');
      motionButton.querySelector('.motion-label').textContent = paused ? 'Resume motion' : 'Pause motion';
      motionButton.querySelector('.motion-icon').textContent = paused ? '▷' : 'Ⅱ';
    }
    if (paused && frame !== null) { cancelAnimationFrame(frame);frame = null;previous = 0; }
    play();
  }
  motionButton?.addEventListener('click', () => setPaused(!paused));
  reduced.addEventListener('change', () => setPaused(reduced.matches));
  document.addEventListener('visibilitychange', play);
  if (hero && finePointer.matches) {
    hero.addEventListener('pointermove', event => {
      if (paused) return;
      const rect = hero.getBoundingClientRect();
      pointer.targetX = Math.max(-1, Math.min(1, (event.clientX - rect.left) / rect.width * 2 - 1));
      pointer.targetY = Math.max(-1, Math.min(1, (event.clientY - rect.top) / rect.height * 2 - 1));
    }, { passive: true });
    hero.addEventListener('pointerleave', () => { pointer.targetX = 0;pointer.targetY = 0; });
  }
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.target === starCanvas) starVisible = entry.isIntersecting;
      if (entry.target === orbCanvas) orbVisible = entry.isIntersecting;
    }
    draw();play();
  }, { threshold: 0 });
  if (starCanvas) observer.observe(starCanvas);
  if (orbCanvas) observer.observe(orbCanvas); else orbVisible = false;
  let resizeTimer;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer);resizeTimer = setTimeout(resize, 100); });
  resize();setPaused(paused);
})();
