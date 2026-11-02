import { useEffect, useRef } from "react";

export const GUIDE_GATHER_MS = 3000;
const FLIGHT_MS = 1500;
export const GUIDE_ARRIVAL_MS = GUIDE_GATHER_MS + FLIGHT_MS;
const DURATION = GUIDE_ARRIVAL_MS + 1200;
const clamp = value => Math.max(0, Math.min(1, value));

// Broad translucent surfaces move through depth, rather than rotating a flat image.
function sheetPoint(u, across, band, time) {
  const taper = Math.sin(u * Math.PI) ** .45;
  const longitude = u * Math.PI * 1.77 + band * 2.1;
  const latitude = .19 * Math.sin(longitude * 1.3 + time * .4 + band) + across * .65 * taper;
  let x = Math.cos(latitude) * Math.cos(longitude);
  let y = Math.sin(latitude);
  let z = Math.cos(latitude) * Math.sin(longitude);
  const pitch = [.75, -1.05, .1][band] + Math.sin(time * .45 + band) * .22;
  [y, z] = [y * Math.cos(pitch) - z * Math.sin(pitch), y * Math.sin(pitch) + z * Math.cos(pitch)];
  const yaw = time * .5 + band * .6;
  [x, z] = [x * Math.cos(yaw) + z * Math.sin(yaw), -x * Math.sin(yaw) + z * Math.cos(yaw)];
  const roll = -.5 + band * .45 + Math.sin(time * .35) * .15;
  return { x: x * Math.cos(roll) - y * Math.sin(roll), y: x * Math.sin(roll) + y * Math.cos(roll), z };
}

export default function GuideSmoke({ anchor }) {
  const canvas = useRef(null);
  useEffect(() => {
    const surface = canvas.current;
    const ctx = surface?.getContext("2d");
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (!ctx || motion.matches) return undefined;
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--glass-focus").trim() || "#197d73";
    ctx.fillRect(0, 0, 1, 1);
    const accent = [...ctx.getImageData(0, 0, 1, 1).data].slice(0, 3);
    const low = Math.min(...accent), range = Math.max(...accent) - low;
    const theme = range ? accent.map(c => 18 + (c - low) / range * 174) : accent;
    const color = (light, alpha) => `rgba(${theme.map(c => Math.round(c + (255 - c) * light)).join(",")},${alpha})`;
    let frame, start, width, height;
    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      const scale = 2;
      surface.width = width * scale;
      surface.height = height * scale;
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
    };
    const location = (time, target) => {
      const progress = clamp((time - GUIDE_GATHER_MS) / FLIGHT_MS);
      const u = progress * progress * (3 - 2 * progress), v = 1 - u;
      const origin = { x: width / 2, y: height / 2 };
      return {
        x: v ** 3 * origin.x + 3 * v * v * u * (origin.x + 80) + 3 * v * u * u * (target.x - 120) + u ** 3 * target.x,
        y: v ** 3 * origin.y + 3 * v * v * u * (origin.y - 130) + 3 * v * u * u * (target.y + 80) + u ** 3 * target.y,
      };
    };
    const render = timestamp => {
      start ??= timestamp;
      const time = timestamp - start;
      ctx.clearRect(0, 0, width, height);
      if (time > DURATION || document.hidden || motion.matches) return;
      const bounds = anchor.current?.getBoundingClientRect();
      if (!bounds) return;
      const target = { x: bounds.left + bounds.width / 2, y: bounds.top + bounds.height / 2 };
      const center = location(time, target);
      const flight = clamp((time - GUIDE_GATHER_MS) / FLIGHT_MS);
      const dissolve = clamp((time - GUIDE_ARRIVAL_MS) / 1200);
      const radius = Math.min(96, width * .18, height * .18) * (1 - flight * .78);
      ctx.globalAlpha = clamp(time / 450) * (1 - dissolve);

      const aura = ctx.createRadialGradient(center.x - radius * .25, center.y - radius * .35, radius * .1, center.x, center.y, radius * 1.12);
      aura.addColorStop(0, color(.98, .12));
      aura.addColorStop(.65, color(.7, .04));
      aura.addColorStop(.88, color(.4, .12));
      aura.addColorStop(1, color(.4, 0));
      ctx.fillStyle = aura;
      ctx.fillRect(center.x - radius * 1.2, center.y - radius * 1.2, radius * 2.4, radius * 2.4);

      const faces = [];
      const project = p => ({ x: center.x + p.x * radius, y: center.y + p.y * radius, z: p.z });
      for (let band = 0; band < 3; band++) {
        let face = { edge: [], rim: [], depths: [], band };
        let previousSide;
        for (let i = 0; i <= 100; i++) {
          const edge = project(sheetPoint(i / 100, -1, band, time / 1000));
          const rim = project(sheetPoint(i / 100, 1, band, time / 1000));
          const depth = (edge.z + rim.z) / 2;
          const side = depth >= 0;
          face.edge.push(edge);
          face.rim.push(rim);
          face.depths.push(depth);
          if (i === 100 || (previousSide !== undefined && side !== previousSide)) {
            faces.push({ ...face, depth: face.depths.reduce((a, b) => a + b, 0) / face.depths.length });
            face = { edge: [edge], rim: [rim], depths: [depth], band };
          }
          previousSide = side;
        }
      }
      faces.sort((a, b) => a.depth - b.depth);
      for (const { edge, rim, depth, band } of faces) {
        const front = (depth + 1) / 2;
        const strength = (band === 2 ? .5 : 1) * (.55 + front * .45);
        const mid = Math.floor(edge.length / 2);
        const gradient = ctx.createLinearGradient(edge[mid].x, edge[mid].y, rim[mid].x, rim[mid].y);
        gradient.addColorStop(0, color(.02, .65 * strength));
        gradient.addColorStop(.13, color(.2, .27 * strength));
        gradient.addColorStop(.55, color(.65, .08 * strength));
        gradient.addColorStop(.88, color(.48, .2 * strength));
        gradient.addColorStop(1, color(.95, .6 * strength));
        ctx.beginPath();
        ctx.moveTo(edge[0].x, edge[0].y);
        edge.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
        rim.slice().reverse().forEach(p => ctx.lineTo(p.x, p.y));
        ctx.closePath();
        ctx.fillStyle = gradient;
        ctx.fill();
        // Draw each continuous surface once, avoiding visible mesh seams.
        ctx.beginPath();
        ctx.moveTo(rim[0].x, rim[0].y);
        rim.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
        ctx.strokeStyle = color(.97, .65 * strength);
        ctx.lineWidth = .9;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      frame = window.requestAnimationFrame(render);
    };
    const stop = () => {
      window.cancelAnimationFrame(frame);
      ctx.clearRect(0, 0, width, height);
    };
    resize();
    frame = window.requestAnimationFrame(render);
    window.addEventListener("resize", resize);
    motion.addEventListener("change", stop);
    return () => {
      stop();
      window.removeEventListener("resize", resize);
      motion.removeEventListener("change", stop);
    };
  }, [anchor]);
  return <>
    <canvas ref={canvas} className="guide-smoke-canvas" aria-hidden="true" />
    <div className="guide-welcome-brand" aria-hidden="true">Nexus Cloud</div>
  </>;
}
