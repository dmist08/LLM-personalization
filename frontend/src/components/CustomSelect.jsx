import { useState, useRef, useEffect } from 'react';

/**
 * CustomSelect — A fully CSS-controlled dropdown that respects dark mode.
 * Replaces native <select> which renders OS-themed popups.
 */
export default function CustomSelect({ options, value, onChange, placeholder, disabled, isDark, id }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  // Close on outside click
  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const selected = options.find(o => o.value === value);

  const bg = isDark ? '#1C1C1C' : '#fff';
  const bgHover = isDark ? '#2a2a2a' : '#f0f4ff';
  const border = isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)';
  const textColor = isDark ? '#F8F9FF' : '#111';
  const mutedColor = isDark ? '#9CA3AF' : '#888';

  return (
    <div ref={ref} id={id} style={{ position: 'relative', display: 'inline-block' }}>
      {/* Trigger button */}
      <button
        type="button"
        onClick={() => !disabled && setOpen(o => !o)}
        disabled={disabled}
        style={{
          height: 30,
          padding: '0 28px 0 10px',
          borderRadius: 8,
          border: `1px solid ${border}`,
          background: isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.04)',
          fontSize: 12,
          fontWeight: 500,
          cursor: disabled ? 'not-allowed' : 'pointer',
          color: selected ? textColor : mutedColor,
          outline: 'none',
          transition: 'border-color 0.15s',
          opacity: disabled ? 0.5 : 1,
          whiteSpace: 'nowrap',
          position: 'relative',
          textAlign: 'left',
          minWidth: 100,
        }}
      >
        {selected?.label || placeholder}
        {/* Chevron */}
        <svg
          width="12" height="12" viewBox="0 0 24 24" fill="none"
          stroke={isDark ? '#aaa' : '#888'} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
          style={{ position: 'absolute', right: 8, top: '50%', transform: `translateY(-50%) rotate(${open ? '180deg' : '0deg'})`, transition: 'transform 0.15s' }}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {/* Dropdown menu */}
      {open && (
        <div
          style={{
            position: 'absolute',
            bottom: '100%',
            left: 0,
            marginBottom: 4,
            minWidth: '100%',
            maxHeight: 220,
            overflowY: 'auto',
            background: bg,
            border: `1px solid ${border}`,
            borderRadius: 10,
            boxShadow: isDark
              ? '0 8px 24px rgba(0,0,0,0.6)'
              : '0 8px 24px rgba(0,0,0,0.12)',
            zIndex: 100,
            padding: '4px 0',
          }}
        >
          {options.map(opt => (
            <button
              key={opt.value}
              type="button"
              onClick={() => { onChange(opt.value); setOpen(false); }}
              style={{
                display: 'block',
                width: '100%',
                padding: '7px 12px',
                fontSize: 12,
                fontWeight: opt.value === value ? 600 : 400,
                color: opt.value === value ? (isDark ? '#3B82F6' : '#2563EB') : textColor,
                background: opt.value === value ? (isDark ? 'rgba(59,130,246,0.12)' : 'rgba(37,99,235,0.08)') : 'transparent',
                border: 'none',
                cursor: 'pointer',
                textAlign: 'left',
                whiteSpace: 'nowrap',
                transition: 'background 0.1s',
              }}
              onMouseEnter={e => { if (opt.value !== value) e.currentTarget.style.background = bgHover; }}
              onMouseLeave={e => { if (opt.value !== value) e.currentTarget.style.background = 'transparent'; }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
