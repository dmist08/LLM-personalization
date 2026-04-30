import { useTheme } from '../context/ThemeContext';

const METHOD_LABELS = {
  no_personalization: 'Base',
  rag_bm25: 'RAG',
  stylevector: 'SV',
  cold_start_sv: 'CS-SV',
};

const METHOD_COLORS = {
  no_personalization: '#9CA3AF',
  rag_bm25: '#F59E0B',
  stylevector: '#10B981',
  cold_start_sv: '#3B82F6',
};

/**
 * MetricsChart — Visual bar chart comparing ROUGE-L scores across methods.
 * Shows horizontal bars with labels and numeric values.
 */
export default function MetricsChart({ metricsMap }) {
  const { isDark } = useTheme();

  if (!metricsMap || Object.keys(metricsMap).length === 0) return null;

  const entries = Object.entries(metricsMap)
    .filter(([, m]) => m && m.rougeL != null)
    .sort((a, b) => b[1].rougeL - a[1].rougeL);

  if (entries.length === 0) return null;

  const maxScore = Math.max(...entries.map(([, m]) => m.rougeL), 0.01);

  return (
    <div
      style={{
        background: isDark ? '#1C1C1C' : '#fff',
        border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.08)'}`,
        borderRadius: 10,
        padding: '16px 20px',
        marginTop: 16,
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
        <span style={{
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: isDark ? '#9CA3AF' : '#6B7280',
        }}>
          ROUGE-L Comparison
        </span>
        <span style={{
          fontSize: 10,
          color: isDark ? '#6B7280' : '#9CA3AF',
        }}>
          vs Ground Truth
        </span>
      </div>

      {/* Bars */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {entries.map(([method, metrics]) => {
          const barWidth = Math.max((metrics.rougeL / maxScore) * 100, 2);
          const color = METHOD_COLORS[method] || '#888';
          const label = METHOD_LABELS[method] || method;
          const isBest = entries[0][0] === method;

          return (
            <div key={method} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              {/* Label */}
              <span style={{
                width: 42,
                fontSize: 11,
                fontWeight: isBest ? 700 : 500,
                color: isBest ? color : (isDark ? '#9CA3AF' : '#6B7280'),
                textAlign: 'right',
                flexShrink: 0,
              }}>
                {label}
              </span>

              {/* Bar container */}
              <div style={{
                flex: 1,
                height: 20,
                background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.03)',
                borderRadius: 4,
                overflow: 'hidden',
                position: 'relative',
              }}>
                <div
                  style={{
                    width: `${barWidth}%`,
                    height: '100%',
                    background: `${color}${isBest ? '' : '88'}`,
                    borderRadius: 4,
                    transition: 'width 0.6s cubic-bezier(0.4,0,0.2,1)',
                    position: 'relative',
                  }}
                >
                  {isBest && (
                    <div style={{
                      position: 'absolute',
                      inset: 0,
                      background: `linear-gradient(90deg, transparent, ${color}33)`,
                      animation: 'shimmerBar 2s infinite',
                    }} />
                  )}
                </div>
              </div>

              {/* Score */}
              <span style={{
                width: 44,
                fontSize: 12,
                fontWeight: isBest ? 700 : 500,
                fontVariantNumeric: 'tabular-nums',
                color: isBest ? color : (isDark ? '#D1D5DB' : '#374151'),
                textAlign: 'right',
                flexShrink: 0,
              }}>
                {(metrics.rougeL * 100).toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>

      {/* Word overlap row */}
      <div style={{
        marginTop: 12,
        paddingTop: 10,
        borderTop: `1px solid ${isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'}`,
        display: 'flex',
        gap: 16,
        flexWrap: 'wrap',
      }}>
        {entries.map(([method, metrics]) => (
          <span key={method} style={{
            fontSize: 10,
            color: isDark ? '#6B7280' : '#9CA3AF',
          }}>
            <span style={{ color: METHOD_COLORS[method], fontWeight: 600 }}>{METHOD_LABELS[method]}</span>
            {' '}overlap: {(metrics.wordOverlap * 100).toFixed(0)}%
          </span>
        ))}
      </div>

      <style>{`@keyframes shimmerBar { 0%,100% { opacity: 0.3; } 50% { opacity: 0.8; } }`}</style>
    </div>
  );
}
