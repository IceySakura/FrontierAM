import { useEffect, useRef, useState } from "react";
import { BookOpen, ExternalLink, X } from "lucide-react";
import { beginGuideVisit, recordGuideVisit } from "./guideWelcome.js";
import GuideSmoke, { GUIDE_ARRIVAL_MS, GUIDE_GATHER_MS } from "./GuideSmoke.jsx";

const GUIDE_URL = window.__NEXUS_DISTRIBUTED_CONSOLE__?.guideUrl || "https://nexus.pages.woa.com/cloud/overview/";

export function UserGuideLink({ className = "", onClick }) {
  return (
    <a className={`user-guide-link ${className}`} href={GUIDE_URL}
      target="_blank" rel="noopener noreferrer" onClick={onClick}>
      <BookOpen size={15} aria-hidden="true" />
      <span>使用指引</span>
      <ExternalLink size={13} aria-hidden="true" />
    </a>
  );
}

export function UserGuideWelcome({ principalId, enabled = false }) {
  const [welcome, setWelcome] = useState(null);
  const [showCard, setShowCard] = useState(false);
  const visited = useRef(false);
  const card = useRef(null);
  const anchor = useRef(null);

  useEffect(() => {
    if (!enabled || !principalId) {
      setWelcome(null);
      return undefined;
    }
    let storage;
    try { storage = window.localStorage; } catch { /* Storage may be disabled. */ }
    const visit = () => {
      if (document.hidden) return;
      if (!visited.current) {
        visited.current = true;
        setWelcome(beginGuideVisit(storage, principalId));
      } else {
        recordGuideVisit(storage, principalId);
      }
    };
    const visibility = () => {
      if (document.hidden) {
        if (visited.current) recordGuideVisit(storage, principalId);
      } else {
        visit();
      }
    };
    visit();
    // Record actual recent use, including a tab left open for several days.
    const timer = window.setInterval(visit, 60000);
    document.addEventListener("visibilitychange", visibility);
    window.addEventListener("pagehide", visit);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", visibility);
      window.removeEventListener("pagehide", visit);
    };
  }, [principalId, enabled]);

  useEffect(() => {
    if (!welcome) return undefined;
    setShowCard(false);
    const reveal = window.setTimeout(() => setShowCard(true),
      window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : GUIDE_ARRIVAL_MS + 200);
    const timer = window.setTimeout(() => {
      if (!card.current?.contains(document.activeElement)) setWelcome(null);
    }, 12000);
    const escape = (event) => {
      if (event.key === "Escape") setWelcome(null);
    };
    window.addEventListener("keydown", escape);
    return () => {
      window.clearTimeout(reveal);
      window.clearTimeout(timer);
      window.removeEventListener("keydown", escape);
    };
  }, [welcome]);

  const dismiss = () => setWelcome(null);
  return (
    <div ref={anchor} className={`guide-welcome-anchor${enabled && welcome ? " is-welcoming" : ""}`}
      style={{ "--guide-arrival": `${GUIDE_ARRIVAL_MS}ms`, "--guide-gather": `${GUIDE_GATHER_MS}ms` }}>
      <UserGuideLink className="icon-button" onClick={dismiss} />
      {enabled && welcome ? (
        <>
          <GuideSmoke anchor={anchor} />
          {showCard ? <aside className="guide-welcome-card" ref={card} aria-label="使用指引提示">
            <button className="guide-welcome-close" type="button" aria-label="关闭欢迎提示" onClick={dismiss}><X size={16} /></button>
            <div role="status">
              <strong>{welcome === "first" ? "欢迎来到 Nexus" : "欢迎回来"}</strong>
              <p>{welcome === "first" ? "从第一个任务开始。环境准备、任务提交和结果查看，都在使用指引里。" : "需要回顾一下操作？使用指引一直在这里，随时可以打开。"}</p>
            </div>
            <a href={GUIDE_URL} target="_blank" rel="noopener noreferrer" onClick={dismiss}>打开使用指引 <span aria-hidden="true">↗</span></a>
          </aside> : null}
        </>
      ) : null}
    </div>
  );
}
