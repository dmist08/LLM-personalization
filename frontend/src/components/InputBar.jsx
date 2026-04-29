import { useState, useEffect, useRef, useCallback } from 'react';
import { getAuthors } from '../services/api';
import { useTheme } from '../context/ThemeContext';
import CustomSelect from './CustomSelect';

const PUBLICATIONS = [
  { code: 'TOI', label: 'The Times of India' },
  { code: 'HT', label: 'Hindustan Times' },
];

const MAX_WORDS = 500;
const MIN_ROWS_HEIGHT = 44;   // px — one line height (compact default)
const MAX_HEIGHT = 260;   // px — cap before scroll kicks in

/** Count words (non-empty token splits) */
const countWords = (text) => text.trim() === '' ? 0 : text.trim().split(/\s+/).length;

export default function InputBar({ onGenerate, isGenerating }) {
  const { isDark } = useTheme();
  const [publication, setPublication] = useState('');
  const [authors, setAuthors] = useState([]);
  const [authorId, setAuthorId] = useState('');
  const [sourceText, setSourceText] = useState('');
  const [loadingAuthors, setLoadingAuthors] = useState(false);

  const textareaRef = useRef(null);

  /* ── Auto-resize textarea ───────────────────────────── */
  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const scrollH = el.scrollHeight;
    el.style.height = Math.min(Math.max(scrollH, MIN_ROWS_HEIGHT), MAX_HEIGHT) + 'px';
    el.style.overflowY = scrollH > MAX_HEIGHT ? 'auto' : 'hidden';
  }, []);

  useEffect(() => { autoResize(); }, [sourceText, autoResize]);

  /* ── Load authors on publication change ─────────────── */
  useEffect(() => {
    if (!publication) { setAuthors([]); setAuthorId(''); return; }
    setLoadingAuthors(true);
    setAuthorId('');
    getAuthors(publication)
      .then(({ authors }) => setAuthors(authors || []))
      .catch(() => setAuthors([]))
      .finally(() => setLoadingAuthors(false));
  }, [publication]);

  /* ── Keyboard shortcut: Ctrl/Cmd + Enter ────────────── */
  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    }
  };

  const wordCount = countWords(sourceText);
  const overLimit = wordCount > MAX_WORDS;
  const canGenerate = publication && authorId && wordCount > 3 && !overLimit && !isGenerating;

  const handleSubmit = () => {
    if (!canGenerate) return;
    const pub = PUBLICATIONS.find(p => p.code === publication);
    const author = authors.find(a => a.id === authorId);
    onGenerate({
      sourceText: sourceText.trim(),
      publication: publication,
      publicationLabel: pub?.label || publication,
      authorId,
      authorName: author?.name || authorId,
    });
  };

  /* ── Build options for CustomSelect ───────────────────── */
  const pubOptions = PUBLICATIONS.map(p => ({ value: p.code, label: p.label }));
  const authorOptions = authors.map(a => ({ value: a.id, label: a.name }));

  /* ── Styles (inline so no Tailwind conflicts) ────────── */
  const wrapperStyle = {
    width: '100%',
    background: 'transparent',
    padding: '8px 0 12px',
  };

  const cardStyle = {
    width: '100%',
    borderRadius: 16,
    border: '1px solid rgba(127,127,127,0.15)',
    boxShadow: '0 8px 32px rgba(0,0,0,0.10)',
    background: 'transparent',
    overflow: 'visible',  // allow dropdown to overflow
  };

  return (
    <div style={wrapperStyle}>
      <div style={cardStyle}>

        {/* ── Top row: Textarea + Generate button ──────────── */}
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: 10, padding: '12px 12px 8px' }}>
          <textarea
            ref={textareaRef}
            value={sourceText}
            onChange={e => setSourceText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Paste source article text here… (Ctrl+Enter to generate)"
            rows={1}
            style={{
              flex: 1,
              minHeight: MIN_ROWS_HEIGHT,
              maxHeight: MAX_HEIGHT,
              resize: 'none',
              overflowY: 'hidden',
              border: 'none',
              outline: 'none',
              background: 'transparent',
              fontSize: 14,
              lineHeight: '22px',
              padding: '10px 12px',
              color: 'inherit',
              fontFamily: 'inherit',
              boxSizing: 'border-box',
              transition: 'height 0.1s ease',
            }}
          />

          {/* Generate button — beside the textarea, bottom-aligned */}
          <button
            id="generate-btn"
            onClick={handleSubmit}
            disabled={!canGenerate}
            title={canGenerate ? 'Generate (Ctrl+Enter)' : 'Fill in all fields first'}
            style={{
              flexShrink: 0,
              height: 40,
              padding: '0 18px',
              borderRadius: 10,
              border: 'none',
              cursor: canGenerate ? 'pointer' : 'not-allowed',
              fontWeight: 600,
              fontSize: 13,
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              marginBottom: 2,
              transition: 'opacity 0.2s, transform 0.1s',
              background: canGenerate
                ? 'linear-gradient(135deg, #004AC6 0%, #2563EB 100%)'
                : 'rgba(127,127,127,0.15)',
              color: canGenerate ? '#fff' : 'rgba(127,127,127,0.7)',
              boxShadow: canGenerate ? '0 4px 14px rgba(37,99,235,0.35)' : 'none',
            }}
            onMouseEnter={e => { if (canGenerate) e.currentTarget.style.transform = 'translateY(-1px)'; }}
            onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; }}
            onMouseDown={e => { if (canGenerate) e.currentTarget.style.transform = 'translateY(1px)'; }}
          >
            {isGenerating ? (
              <>
                <span
                  className="material-symbols-outlined"
                  style={{ fontSize: 17, animation: 'spin 1s linear infinite' }}
                >
                  progress_activity
                </span>
                Generating…
              </>
            ) : (
              <>
                <span className="material-symbols-outlined" style={{ fontSize: 17 }}>
                  auto_awesome
                </span>
                Generate
              </>
            )}
          </button>
        </div>

        {/* ── Divider ──────────────────────────────────────── */}
        <div style={{ height: 1, background: 'rgba(127,127,127,0.10)', margin: '0 12px' }} />

        {/* ── Bottom row: Source · Author · Word count ─────── */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '8px 12px 10px',
          flexWrap: 'wrap',
        }}>
          {/* Source dropdown */}
          <CustomSelect
            id="source-select"
            options={pubOptions}
            value={publication}
            onChange={setPublication}
            placeholder="Source…"
            isDark={isDark}
          />

          {/* Author dropdown */}
          <CustomSelect
            id="author-select"
            options={authorOptions}
            value={authorId}
            onChange={setAuthorId}
            placeholder={loadingAuthors ? 'Loading…' : 'Author…'}
            disabled={!publication || loadingAuthors}
            isDark={isDark}
          />

          {/* Spacer */}
          <span style={{ flex: 1 }} />

          {/* Word count */}
          <span style={{
            fontSize: 12,
            fontWeight: 500,
            letterSpacing: '0.02em',
            color: overLimit ? '#ef4444' : wordCount > MAX_WORDS * 0.85 ? '#f59e0b' : 'rgba(127,127,127,0.7)',
            transition: 'color 0.2s',
            whiteSpace: 'nowrap',
          }}>
            {wordCount} / {MAX_WORDS} words
          </span>
        </div>
      </div>

      {/* Spin keyframe injected once */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
