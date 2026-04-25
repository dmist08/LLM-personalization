import { useState, useEffect } from 'react';
import { getAuthors } from '../services/api';

const PUBLICATIONS = [
  { code: 'TOI',  label: 'The Times of India' },
  { code: 'HINDU', label: 'The Hindu' },
  { code: 'IE',   label: 'Indian Express' },
  { code: 'HT',   label: 'Hindustan Times' },
  { code: 'MINT', label: 'Mint' },
  { code: 'NDTV', label: 'NDTV' },
  { code: 'WIRE', label: 'The Wire' },
];

const MAX_CHARS = 5000;

export default function InputBar({ onGenerate, isGenerating }) {
  const [publication, setPublication] = useState('');
  const [authors, setAuthors]     = useState([]);
  const [authorId, setAuthorId]   = useState('');
  const [sourceText, setSourceText] = useState('');
  const [loadingAuthors, setLoadingAuthors] = useState(false);

  // Load authors when publication changes
  useEffect(() => {
    if (!publication) {
      setAuthors([]);
      setAuthorId('');
      return;
    }
    setLoadingAuthors(true);
    setAuthorId('');
    getAuthors(publication)
      .then(({ authors }) => setAuthors(authors || []))
      .catch(() => setAuthors([]))
      .finally(() => setLoadingAuthors(false));
  }, [publication]);

  const canGenerate = publication && authorId && sourceText.trim().length > 20 && !isGenerating;

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

  return (
    <div className="absolute bottom-0 left-0 w-full p-6 md:p-8 bg-gradient-to-t from-surface dark:from-[#0F0F0F] via-surface/90 dark:via-[#0F0F0F]/90 to-transparent">
      <div className="max-w-3xl mx-auto bg-surface-container-lowest dark:bg-[#1C1C1C] rounded-xl shadow-[0_12px_32px_rgba(21,28,37,0.06)] dark:shadow-[0_12px_32px_rgba(0,0,0,0.4)] border border-outline-variant/10 overflow-hidden flex flex-col">

        {/* Row 1: Controls */}
        <div className="flex items-center gap-3 p-4 border-b border-outline-variant/10 bg-surface-container-low/50 dark:bg-[#171717]/50 flex-wrap">
          {/* Publication selector */}
          <select
            value={publication}
            onChange={e => setPublication(e.target.value)}
            className="w-[155px] h-9 text-[13px] font-medium bg-surface dark:bg-[#1C1C1C] text-on-surface dark:text-[#F8F9FF] border-none rounded-md shadow-sm focus:ring-1 focus:ring-primary appearance-none cursor-pointer px-3"
          >
            <option value="">Select Source...</option>
            {PUBLICATIONS.map(p => (
              <option key={p.code} value={p.code}>{p.label}</option>
            ))}
          </select>

          {/* Author selector */}
          <select
            value={authorId}
            onChange={e => setAuthorId(e.target.value)}
            disabled={!publication || loadingAuthors}
            className="w-[175px] h-9 text-[13px] font-medium bg-surface dark:bg-[#1C1C1C] text-on-surface dark:text-[#F8F9FF] border-none rounded-md shadow-sm focus:ring-1 focus:ring-primary appearance-none cursor-pointer px-3 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <option value="">
              {loadingAuthors ? 'Loading...' : 'Select Author...'}
            </option>
            {authors.map(a => (
              <option key={a.id} value={a.id}>{a.name}</option>
            ))}
          </select>

          {/* Selected journalist chip */}
          {authorId && (
            <span className="px-3 py-1 bg-[#DBEAFE] dark:bg-[#1E3A5F] text-[#004AC6] dark:text-[#3B82F6] text-[12px] font-medium rounded-full">
              ✦ Style vector loaded
            </span>
          )}
        </div>

        {/* Row 2: Textarea */}
        <textarea
          value={sourceText}
          onChange={e => setSourceText(e.target.value.slice(0, MAX_CHARS))}
          placeholder="Paste source article text here to begin style vector extraction..."
          className="w-full h-24 p-4 text-[14px] bg-transparent border-none focus:ring-0 resize-none text-on-surface dark:text-[#F8F9FF] placeholder:text-outline dark:placeholder:text-[#555]"
        />

        {/* Row 3: Actions */}
        <div className="flex items-center justify-between p-4 bg-surface-container-low/30 dark:bg-[#171717]/30">
          <span className={`text-[12px] font-medium tracking-wide ${
            sourceText.length > MAX_CHARS * 0.9 ? 'text-error' : 'text-outline'
          }`}>
            {sourceText.length} / {MAX_CHARS}
          </span>

          <button
            onClick={handleSubmit}
            disabled={!canGenerate}
            className={`px-5 py-2 rounded-md font-medium text-[13px] flex items-center gap-2 transition-colors ${
              canGenerate
                ? 'bg-gradient-to-br from-[#004AC6] to-[#2563EB] text-white hover:brightness-110 active:scale-95'
                : 'bg-surface-variant dark:bg-[#2a2a2a] text-outline cursor-not-allowed'
            }`}
          >
            {isGenerating ? (
              <>
                <span className="material-symbols-outlined animate-spin-slow" style={{ fontSize: 18 }}>
                  progress_activity
                </span>
                Generating...
              </>
            ) : (
              <>
                <span className="material-symbols-outlined" style={{ fontSize: 18 }}>auto_awesome</span>
                Generate Headlines
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
