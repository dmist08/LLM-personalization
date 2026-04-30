/**
 * MetricsChart — ROUGE-L horizontal bar chart
 * Matches the design in the uploaded reference image.
 */

const METHOD_META = {
  stylevector: { label: 'SV', color: '#22C55E' },  // green
  lora_finetuned: { label: 'lora_finetuned', color: '#94A3B8' }, // slate
  no_personalization: { label: 'Base', color: '#94A3B8' },  // slate
  rag_bm25: { label: 'RAG', color: '#FBBF24' },  // amber
  cold_start_sv: { label: 'CS-SV', color: '#818CF8' },  // indigo
};

export default function MetricsChart({ metricsMap }) {
  if (!metricsMap || Object.keys(metricsMap).length === 0) return null;

  // Build rows sorted by rougeL descending
  const rows = Object.entries(METHOD_META)
    .filter(([key]) => metricsMap[key])
    .map(([key, meta]) => ({
      key,
      label: meta.label,
      color: meta.color,
      rougeL: metricsMap[key]?.rougeL ?? 0,
      overlap: metricsMap[key]?.overlap ?? 0,
    }))
    .sort((a, b) => b.rougeL - a.rougeL);

  if (rows.length === 0) return null;

  const maxRouge = Math.max(...rows.map(r => r.rougeL), 0.001);

  return (
    <div className="mt-6 bg-surface-container-lowest dark:bg-[#1C1C1C] border border-surface-variant/30 dark:border-[#2a2a2a] rounded-[8px] p-5">

      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <span className="text-[10px] font-bold tracking-[0.1em] uppercase text-on-surface dark:text-[#F8F9FF]">
          ROUGE-L Comparison
        </span>
        <span className="text-[11px] text-on-surface-variant dark:text-[#9CA3AF]">
          vs Ground Truth
        </span>
      </div>

      {/* Bars */}
      <div className="flex flex-col gap-3">
        {rows.map(row => {
          const pct = maxRouge > 0 ? (row.rougeL / maxRouge) * 100 : 0;
          const isTop = row.key === rows[0].key;

          return (
            <div key={row.key} className="flex items-center gap-3">
              {/* Label */}
              <div className="w-[100px] flex-shrink-0 text-right">
                <span className={`text-[12px] font-medium ${isTop
                    ? 'font-semibold'
                    : 'text-on-surface-variant dark:text-[#9CA3AF]'
                  }`}
                  style={{ color: isTop ? row.color : undefined }}
                >
                  {row.label}
                </span>
              </div>

              {/* Bar track */}
              <div className="flex-1 bg-surface-container-high dark:bg-[#2a2a2a] rounded-full h-[10px] overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700 ease-out"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: row.color,
                    opacity: isTop ? 1 : 0.75,
                  }}
                />
              </div>

              {/* Score */}
              <div className="w-[42px] flex-shrink-0 text-right">
                <span className={`text-[12px] font-semibold tabular-nums ${isTop ? '' : 'text-on-surface-variant dark:text-[#9CA3AF]'
                  }`}
                  style={{ color: isTop ? row.color : undefined }}
                >
                  {(row.rougeL * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer — token overlap legend */}
      <div className="mt-5 pt-4 border-t border-surface-variant/20 dark:border-[#2a2a2a] flex flex-wrap gap-x-4 gap-y-1">
        {rows.map(row => (
          <span key={row.key} className="text-[11px]">
            <span className="font-semibold" style={{ color: row.color }}>
              {row.label}
            </span>
            <span className="text-on-surface-variant dark:text-[#9CA3AF]">
              {' '}overlap: {row.overlap}%
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}