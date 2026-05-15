'use client';

import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// ── Types ─────────────────────────────────────────────────────────────────────

interface Mode { name: string; description: string; vulnerable: boolean; }
interface Source { title: string; url: string; score: number; published_date: string; }
interface ResearchResult {
  query: string; report: string; thinking: string;
  sources: Source[]; search_count: number; system_prompt_name: string; timestamp: string;
}
interface Action { kind: string; value: string; }

// ── Logo ──────────────────────────────────────────────────────────────────────

function Logo({ size = 36 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 44 44" fill="none" aria-label="Deep Research Agent">
      <defs>
        <linearGradient id="lg" x1="0" y1="0" x2="44" y2="44" gradientUnits="userSpaceOnUse">
          <stop stopColor="#818cf8" />
          <stop offset="1" stopColor="#c084fc" />
        </linearGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="1.5" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>
      {/* Lens ring */}
      <circle cx="19" cy="19" r="13.5" stroke="url(#lg)" strokeWidth="2.2" filter="url(#glow)" />
      {/* Neural nodes */}
      <circle cx="19" cy="12" r="2.4" fill="url(#lg)" filter="url(#glow)" />
      <circle cx="12.5" cy="22" r="2.4" fill="url(#lg)" filter="url(#glow)" />
      <circle cx="25.5" cy="22" r="2.4" fill="url(#lg)" filter="url(#glow)" />
      {/* Connections */}
      <line x1="19" y1="12" x2="12.5" y2="22" stroke="url(#lg)" strokeWidth="1.3" opacity="0.55" />
      <line x1="19" y1="12" x2="25.5" y2="22" stroke="url(#lg)" strokeWidth="1.3" opacity="0.55" />
      <line x1="12.5" y1="22" x2="25.5" y2="22" stroke="url(#lg)" strokeWidth="1.3" opacity="0.55" />
      {/* Handle */}
      <line x1="29.5" y1="29.5" x2="38.5" y2="38.5" stroke="url(#lg)" strokeWidth="3.8"
        strokeLinecap="round" filter="url(#glow)" />
    </svg>
  );
}

// ── SSE parser ────────────────────────────────────────────────────────────────

function parseSSE(text: string): { event: string; data: unknown }[] {
  return text.split('\n\n').filter(p => p.trim()).flatMap(block => {
    let event = 'message', raw = '';
    for (const line of block.split('\n')) {
      if (line.startsWith('event: ')) event = line.slice(7).trim();
      if (line.startsWith('data: ')) raw = line.slice(6).trim();
    }
    if (!raw) return [];
    try { return [{ event, data: JSON.parse(raw) }]; } catch { return []; }
  });
}

// ── Mode selector (sidebar) ───────────────────────────────────────────────────

const MODE_ICONS: Record<string, string> = {
  general: '◈', tech: '⌬', market: '◎', science: '✦', news: '◉',
  legal: '▣', finance_advisor_legacy: '◈', customer_support_v1: '◎',
  hr_assistant_v2: '▣', legal_assistant: '▣',
};

function ModeList({ modes, selected, onSelect }: {
  modes: Mode[]; selected: string; onSelect: (n: string) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      {modes.map(m => {
        const active = m.name === selected;
        return (
          <button
            key={m.name}
            onClick={() => onSelect(m.name)}
            style={{
              background: active ? 'linear-gradient(135deg,rgba(99,102,241,0.18),rgba(168,85,247,0.12))' : 'transparent',
              border: `1px solid ${active ? 'rgba(99,102,241,0.4)' : 'transparent'}`,
              borderRadius: '8px',
              padding: '8px 10px',
              textAlign: 'left',
              cursor: 'pointer',
              transition: 'all 0.15s',
            }}
          >
            <div className="flex items-center gap-2">
              <span style={{ fontSize: '0.7rem', color: m.vulnerable ? 'var(--danger)' : active ? '#a5b4fc' : 'var(--text-3)' }}>
                {m.vulnerable ? '⬟' : (MODE_ICONS[m.name] || '◈')}
              </span>
              <span style={{
                fontSize: '0.82rem', fontWeight: active ? 600 : 400,
                color: m.vulnerable ? '#fb7185' : active ? 'var(--text)' : 'var(--text-2)',
              }}>
                {m.name}
              </span>
              {m.vulnerable && (
                <span style={{ fontSize: '0.6rem', background: 'var(--danger-bg)', color: 'var(--danger)', border: '1px solid rgba(244,63,94,0.3)', borderRadius: '4px', padding: '1px 5px', marginLeft: 'auto' }}>
                  VULN
                </span>
              )}
            </div>
            {active && m.description && (
              <p style={{ fontSize: '0.72rem', color: 'var(--text-3)', marginTop: '4px', marginLeft: '20px', lineHeight: 1.4 }}>
                {m.description}
              </p>
            )}
          </button>
        );
      })}
    </div>
  );
}

// ── Thinking panel ────────────────────────────────────────────────────────────

function ThinkingPanel({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(true);
  const bodyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (live && bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  }, [text, live]);

  if (!text && !live) return null;

  return (
    <div className="animate-fade-in" style={{
      background: 'var(--think-bg)',
      border: `1px solid var(--think-border)`,
      borderRadius: '12px',
      overflow: 'hidden',
      boxShadow: '0 0 20px rgba(139,92,246,0.08)',
    }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: '8px',
          width: '100%', padding: '10px 14px',
          background: 'none', border: 'none', cursor: 'pointer',
          borderBottom: open ? '1px solid var(--think-border)' : 'none',
        }}
      >
        <span style={{ fontSize: '0.95rem' }}>🧠</span>
        <span style={{ fontSize: '0.82rem', fontWeight: 600, color: '#a78bfa', flex: 1, textAlign: 'left' }}>
          Model reasoning {live && <span className="cursor" />}
        </span>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-3)', transform: open ? 'none' : 'rotate(180deg)', transition: 'transform 0.2s', display: 'inline-block' }}>▼</span>
      </button>
      {open && (
        <div
          ref={bodyRef}
          style={{
            maxHeight: '260px', overflowY: 'auto', padding: '12px 14px',
            fontFamily: "'JetBrains Mono','Fira Code',monospace",
            fontSize: '0.75rem', lineHeight: 1.65, color: 'var(--think-text)',
            whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          }}
        >
          {text || <span style={{ color: 'var(--text-3)', fontStyle: 'italic' }}>Waiting for model to begin reasoning…</span>}
        </div>
      )}
    </div>
  );
}

// ── Action log ────────────────────────────────────────────────────────────────

function ActionLog({ actions }: { actions: Action[] }) {
  if (!actions.length) return null;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      {actions.map((a, i) => (
        <div key={i} className="animate-fade-in" style={{
          display: 'flex', alignItems: 'center', gap: '8px',
          padding: '7px 12px', borderRadius: '8px',
          background: 'var(--surface-2)', border: '1px solid var(--border)',
          fontSize: '0.78rem',
        }}>
          <span style={{ fontSize: '0.85rem' }}>{a.kind === 'search' ? '🔎' : '📄'}</span>
          <span style={{ color: 'var(--text-3)', textTransform: 'uppercase', fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.08em', minWidth: '40px' }}>{a.kind}</span>
          <span style={{ color: 'var(--text-2)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{a.value}</span>
        </div>
      ))}
    </div>
  );
}

// ── Metrics bar ───────────────────────────────────────────────────────────────

function MetricsBar({ result }: { result: ResearchResult }) {
  const items = [
    { icon: '🔎', label: 'Searches', value: String(result.search_count) },
    { icon: '📚', label: 'Sources', value: String(result.sources.length) },
    { icon: '◈', label: 'Mode', value: result.system_prompt_name },
  ];
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '10px' }}>
      {items.map(m => (
        <div key={m.label} style={{
          padding: '14px 16px', borderRadius: '12px',
          background: 'var(--surface-2)', border: '1px solid var(--border)',
          textAlign: 'center',
        }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text)', lineHeight: 1.1 }}>{m.value}</div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-3)', marginTop: '4px', textTransform: 'uppercase', letterSpacing: '0.07em' }}>{m.label}</div>
        </div>
      ))}
    </div>
  );
}

// ── Sources panel ─────────────────────────────────────────────────────────────

function SourcesPanel({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  if (!sources.length) return null;
  const sorted = [...sources].sort((a, b) => b.score - a.score);

  return (
    <div style={{ borderRadius: '12px', border: '1px solid var(--border)', overflow: 'hidden', background: 'var(--surface-2)' }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          width: '100%', padding: '11px 16px',
          background: 'none', border: 'none', cursor: 'pointer',
          borderBottom: open ? '1px solid var(--border)' : 'none',
        }}
      >
        <span style={{ fontSize: '0.84rem', fontWeight: 600, color: 'var(--text)' }}>📚 Sources <span style={{ color: 'var(--text-3)', fontWeight: 400 }}>({sources.length})</span></span>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-3)', transform: open ? 'none' : 'rotate(180deg)', transition: 'transform 0.2s', display: 'inline-block' }}>▼</span>
      </button>
      {open && (
        <ol style={{ padding: '10px 16px 14px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {sorted.map((s, i) => (
            <li key={i} style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-3)', minWidth: '16px' }}>{i + 1}.</span>
              <div>
                <span style={{ fontSize: '0.83rem', color: 'var(--text)', fontWeight: 500 }}>{s.title || s.url}</span>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-3)', marginLeft: '8px' }}>
                  {s.score.toFixed(2)}{s.published_date ? ` · ${s.published_date}` : ''}
                </span>
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Page() {
  const [modes, setModes] = useState<Mode[]>([]);
  const [query, setQuery] = useState('');
  const [selectedMode, setSelectedMode] = useState('general');
  const [running, setRunning] = useState(false);
  const [thinking, setThinking] = useState('');
  const [actions, setActions] = useState<Action[]>([]);
  const [result, setResult] = useState<ResearchResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/modes')
      .then(r => r.json())
      .then(({ modes, default: def }: { modes: Mode[]; default: string }) => {
        setModes(modes);
        setSelectedMode(def);
      })
      .catch(() => {});
  }, []);

  const run = async () => {
    if (!query.trim() || running) return;
    setRunning(true); setThinking(''); setActions([]); setResult(null); setError(null);

    try {
      const resp = await fetch('/api/research', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), mode: selectedMode }),
      });
      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const cut = buf.lastIndexOf('\n\n');
        if (cut === -1) continue;
        const chunk = buf.slice(0, cut + 2);
        buf = buf.slice(cut + 2);

        for (const { event, data } of parseSSE(chunk)) {
          const d = data as Record<string, unknown>;
          if (event === 'thinking')  setThinking(p => p + (d.token as string));
          if (event === 'action')    setActions(p => [...p, { kind: d.kind as string, value: d.value as string }]);
          if (event === 'complete')  setResult(data as ResearchResult);
          if (event === 'error')     setError(d.message as string);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  const clear = () => { setQuery(''); setThinking(''); setActions([]); setResult(null); setError(null); };

  const hasContent = running || result || error;

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>

      {/* ── Sidebar ─────────────────────────────────────── */}
      <aside style={{
        width: '240px', flexShrink: 0, display: 'flex', flexDirection: 'column',
        borderRight: '1px solid var(--border)', background: 'var(--surface)',
        overflow: 'hidden',
      }}>
        {/* Brand */}
        <div style={{
          padding: '20px 16px 18px',
          borderBottom: '1px solid var(--border)',
          background: 'linear-gradient(180deg, rgba(99,102,241,0.06) 0%, transparent 100%)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Logo size={34} />
            <div>
              <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text)', lineHeight: 1.1 }}>Deep Research</div>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-3)', marginTop: '2px', letterSpacing: '0.05em' }}>AGENT</div>
            </div>
          </div>
        </div>

        {/* Mode list */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '12px 10px' }}>
          <p style={{ fontSize: '0.65rem', color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 600, marginBottom: '8px', paddingLeft: '4px' }}>
            Research Mode
          </p>
          <ModeList modes={modes} selected={selectedMode} onSelect={setSelectedMode} />
        </div>

        {/* Footer */}
        <div style={{ padding: '12px 16px', borderTop: '1px solid var(--border)' }}>
          <p style={{ fontSize: '0.68rem', color: 'var(--text-3)', lineHeight: 1.5 }}>
            Powered by local LLM<br />+ local datasets
          </p>
        </div>
      </aside>

      {/* ── Main ────────────────────────────────────────── */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--bg)' }}>

        {/* Header */}
        <header style={{
          padding: '16px 28px', borderBottom: '1px solid var(--border)',
          background: 'var(--surface)',
          display: 'flex', alignItems: 'center', gap: '12px',
        }}>
          <div>
            <h1 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text)', margin: 0 }}>Research</h1>
            <p style={{ fontSize: '0.74rem', color: 'var(--text-3)', margin: 0, marginTop: '1px' }}>
              Ask a question — the agent searches datasets and synthesises a structured report
            </p>
          </div>
          {running && (
            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '7px' }}>
              <span style={{
                width: '7px', height: '7px', borderRadius: '50%', background: 'var(--accent)',
                display: 'inline-block', animation: 'pulse-ring 1.5s ease infinite',
              }} />
              <span style={{ fontSize: '0.78rem', color: 'var(--text-2)' }} className="shimmer">Researching…</span>
            </div>
          )}
        </header>

        {/* Content */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '24px 28px', display: 'flex', flexDirection: 'column', gap: '20px' }}>

          {/* Query input */}
          <div style={{
            borderRadius: '14px', border: '1px solid var(--border-2)',
            background: 'var(--surface-2)', overflow: 'hidden',
            transition: 'border-color 0.15s',
          }}>
            <textarea
              style={{
                display: 'block', width: '100%', minHeight: '80px',
                padding: '14px 16px', resize: 'none',
                background: 'transparent', border: 'none', outline: 'none',
                color: 'var(--text)', fontSize: '0.92rem', lineHeight: 1.6,
                fontFamily: 'inherit',
              }}
              placeholder="What would you like to research? (⌘+Enter to run)"
              value={query}
              disabled={running}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') run(); }}
            />
            <div style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              padding: '8px 10px', borderTop: '1px solid var(--border)',
              background: 'var(--surface)',
            }}>
              <button
                onClick={run}
                disabled={running || !query.trim()}
                style={{
                  padding: '7px 18px', borderRadius: '8px', border: 'none', cursor: 'pointer',
                  fontWeight: 600, fontSize: '0.82rem', transition: 'opacity 0.15s',
                  background: 'linear-gradient(135deg, var(--accent), var(--accent-2))',
                  color: '#fff', opacity: (running || !query.trim()) ? 0.4 : 1,
                  boxShadow: (!running && query.trim()) ? '0 0 16px rgba(99,102,241,0.35)' : 'none',
                }}
              >
                {running ? '⏳  Running…' : '🚀  Research'}
              </button>
              {hasContent && (
                <button
                  onClick={clear}
                  disabled={running}
                  style={{
                    padding: '7px 14px', borderRadius: '8px', cursor: 'pointer',
                    background: 'transparent', border: '1px solid var(--border-2)',
                    color: 'var(--text-3)', fontSize: '0.82rem', transition: 'all 0.15s',
                    opacity: running ? 0.4 : 1,
                  }}
                >
                  ✕  Clear
                </button>
              )}
              <span style={{ marginLeft: 'auto', fontSize: '0.72rem', color: 'var(--text-3)' }}>
                Mode: <span style={{ color: 'var(--text-2)' }}>{selectedMode}</span>
              </span>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="animate-fade-in" style={{
              padding: '12px 16px', borderRadius: '10px',
              background: 'var(--danger-bg)', border: '1px solid rgba(244,63,94,0.3)',
              color: '#fda4af', fontSize: '0.85rem',
            }}>
              ❌ {error}
            </div>
          )}

          {/* Live: thinking + actions */}
          {running && (
            <>
              <ThinkingPanel text={thinking} live />
              <ActionLog actions={actions} />
            </>
          )}

          {/* Result */}
          {result && !running && (
            <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
              <MetricsBar result={result} />

              <div style={{ borderTop: '1px solid var(--border-2)', paddingTop: '4px' }} />

              {result.thinking && <ThinkingPanel text={result.thinking} live={false} />}

              <div className="prose">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.report}</ReactMarkdown>
              </div>

              <SourcesPanel sources={result.sources} />

              <a
                download={`report_${result.query.slice(0, 32).replace(/\s+/g, '_')}.md`}
                href={`data:text/markdown;charset=utf-8,${encodeURIComponent(result.report)}`}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: '6px',
                  alignSelf: 'flex-start', padding: '8px 16px',
                  borderRadius: '8px', border: '1px solid var(--border-2)',
                  color: 'var(--text-2)', fontSize: '0.82rem', textDecoration: 'none',
                  transition: 'border-color 0.15s, color 0.15s',
                }}
              >
                ⬇ Download Report (.md)
              </a>
            </div>
          )}

          {/* Empty state */}
          {!hasContent && (
            <div style={{
              flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
              justifyContent: 'center', gap: '12px', paddingTop: '60px', opacity: 0.4,
            }}>
              <Logo size={48} />
              <p style={{ fontSize: '0.85rem', color: 'var(--text-3)', textAlign: 'center', lineHeight: 1.6 }}>
                Enter a research question above<br />and select a mode from the sidebar
              </p>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}
