import { useState } from 'react';

const CARD_CONFIG = {
  no_personalization: {
    label: 'NO PERSONALIZATION',
    badge: null,
    accent: false,
    featured: false,
    description: 'Generic LLM output without any style conditioning.',
  },
  rag_bm25: {
    label: 'RAG (BM25)',
    badge: null,
    accent: false,
    featured: false,
    description: 'Retrieval-Augmented Generation using BM25 keyword search.',
  },
  stylevector: {
    label: 'STYLEVECTOR',
    badge: null,
    accent: true,
    featured: false,
    description: 'Standard StyleVector with full author history available.',
  },
  cold_start_sv: {
    label: 'COLD-START SV (OURS)',
    badge: '★ Novel',
    accent: true,
    featured: true,
    description: 'Our novel Cold-Start approach with activation steering — no prior author data needed.',
  },
};

function SkeletonCard({ config }) {
  return (
    <div className={`bg-surface-container-lowest dark:bg-[#1C1C1C] rounded-[8px] p-5 border relative overflow-hidden ${
      config.featured
        ? 'border-[#2563EB]/30 shadow-[0_4px_16px_rgba(37,99,235,0.12)]'
        : 'border-surface-variant/30 dark:border-[#2a2a2a]'
    }`}>
      {config.featured && (
        <div className="absolute inset-0 bg-primary/5 dark:bg-[#1E3A5F]/20" />
      )}
      <div className="flex justify-between items-center mb-3 relative z-10">
        <span className={`text-[10px] font-bold tracking-[0.08em] uppercase ${
          config.featured || config.accent
            ? 'text-primary dark:text-[#3B82F6]'
            : 'text-on-surface-variant dark:text-[#9CA3AF]'
        }`}>
          {config.label}
        </span>
        {config.featured && (
          <span className="flex items-center gap-1 text-[10px] font-medium text-primary dark:text-[#3B82F6] bg-primary-container/10 dark:bg-[#1E3A5F] px-1.5 py-0.5 rounded">
            <span className="material-symbols-outlined animate-spin-slow" style={{ fontSize: 12 }}>autorenew</span>
            Generating
          </span>
        )}
      </div>
      <div className="space-y-2.5 mt-2 relative z-10">
        <div className="h-4 bg-surface-container-high dark:bg-[#2a2a2a] rounded w-full animate-shimmer" />
        <div className="h-4 bg-surface-container-high dark:bg-[#2a2a2a] rounded w-5/6 animate-shimmer" />
        <div className="h-4 bg-surface-container-high dark:bg-[#2a2a2a] rounded w-2/3 animate-shimmer" />
      </div>
    </div>
  );
}

export default function HeadlineCard({ type, headline, rougeL, latencyMs, isLoading, isCopied, onCopy }) {
  const config = CARD_CONFIG[type] || CARD_CONFIG.no_personalization;
  const [showTooltip, setShowTooltip] = useState(false);

  if (isLoading) return <SkeletonCard config={config} />;

  return (
    <div
      className={`bg-surface-container-lowest dark:bg-[#1C1C1C] rounded-[8px] p-5 relative overflow-hidden transition-shadow hover:shadow-[0_12px_32px_rgba(21,28,37,0.06)] dark:hover:shadow-[0_12px_32px_rgba(0,0,0,0.4)] ${
        config.featured
          ? 'border border-[#2563EB]/30 shadow-[0_4px_16px_rgba(37,99,235,0.12)] border-l-[3px] border-l-primary'
          : 'border border-surface-variant/30 dark:border-[#2a2a2a] shadow-[0_2px_8px_rgba(21,28,37,0.04)]'
      }`}
    >
      {config.featured && (
        <div className="absolute inset-0 bg-primary/5 dark:bg-[#1E3A5F]/20 pointer-events-none" />
      )}

      {/* Card Header */}
      <div className="flex justify-between items-center mb-3 relative z-10">
        <div className="flex items-center gap-2">
          <span className={`text-[10px] font-bold tracking-[0.08em] uppercase ${
            config.featured || config.accent
              ? 'text-primary dark:text-[#3B82F6]'
              : 'text-on-surface-variant dark:text-[#9CA3AF]'
          }`}>
            {config.label}
          </span>
          {config.badge && (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded-sm bg-[#DBEAFE] dark:bg-[#1E3A5F] text-[#004AC6] dark:text-[#3B82F6] text-[9px] font-bold uppercase tracking-wider">
              {config.badge}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* ROUGE-L score (only for cold_start_sv) */}
          {rougeL !== undefined && (
            <span className="text-[11px] text-[#2563EB] dark:text-[#3B82F6] font-bold uppercase tracking-widest">
              ROUGE-L {rougeL}
            </span>
          )}
          {/* Latency */}
          {latencyMs !== undefined && (
            <span className="text-[10px] text-outline px-1.5 py-0.5 bg-surface-container-high dark:bg-[#2a2a2a] rounded">
              {(latencyMs / 1000).toFixed(1)}s
            </span>
          )}
          {/* Copy button */}
          <div className="relative">
            <button
              onClick={() => onCopy(headline)}
              onMouseEnter={() => setShowTooltip(true)}
              onMouseLeave={() => setShowTooltip(false)}
              className="text-outline hover:text-on-surface dark:hover:text-[#F8F9FF] transition-colors"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
                {isCopied ? 'check' : 'content_copy'}
              </span>
            </button>
            {showTooltip && (
              <span className="absolute -top-7 right-0 text-[11px] bg-[#111] text-white px-2 py-1 rounded whitespace-nowrap">
                {isCopied ? 'Copied!' : 'Copy'}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Headline Text */}
      <h3 className={`text-[17px] tracking-[-0.02em] leading-snug relative z-10 ${
        config.featured
          ? 'font-semibold text-[#111111] dark:text-[#F8F9FF]'
          : 'font-medium text-[#111111] dark:text-[#F8F9FF]'
      }`}>
        {headline}
      </h3>

      {/* Description (hover info) */}
      <p className="text-[11px] text-on-surface-variant dark:text-[#9CA3AF] mt-3 relative z-10">
        {config.description}
      </p>
    </div>
  );
}
