import { useNavigate } from 'react-router-dom';
import { useState } from 'react';
import Sidebar from '../components/Sidebar';
import InputBar from '../components/InputBar';
import { useTheme } from '../context/ThemeContext';
import { generateHeadlines } from '../services/api';
import { useAuth } from '../context/AuthContext';

const FEATURE_CHIPS = [
  'Activation Steering',
  'Cold-Start Clustering',
  '43 Indian Journalists',
];

export default function HomePage() {
  const { isDark, toggleTheme } = useTheme();
  const { user } = useAuth();
  const navigate = useNavigate();
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null);

  const handleGenerate = async (payload) => {
    setIsGenerating(true);
    setError(null);
    try {
      const data = await generateHeadlines({ ...payload, sessionId: null });
      // Navigate to chat page with results embedded in state
      navigate(`/chat/${data.session_id}`, { state: { results: data, payload } });
    } catch (err) {
      setError(err.response?.data?.error || 'Generation failed. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className={`${isDark ? 'dark' : ''}`}>
      <div className="bg-surface dark:bg-[#0F0F0F] text-on-surface dark:text-[#F8F9FF] h-screen flex overflow-hidden transition-colors duration-300">

        <Sidebar onNewChat={() => {}} />

        <main className="flex-1 md:ml-[260px] h-full flex flex-col relative">

          {/* Top header */}
          <header className="h-16 items-center justify-between px-8 bg-surface/80 dark:bg-[#0F0F0F]/80 backdrop-blur-md sticky top-0 z-30 hidden md:flex">
            <h1 className="text-[17px] font-medium tracking-[-0.02em] text-on-surface dark:text-[#F8F9FF]">
              Cold-Start StyleVector
            </h1>
            <div className="flex items-center gap-6 text-sm text-on-surface-variant dark:text-[#9CA3AF] font-medium">
              <a href="https://arxiv.org" target="_blank" rel="noreferrer"
                className="hover:text-primary dark:hover:text-[#3B82F6] transition-colors flex items-center gap-1">
                Paper
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>north_east</span>
              </a>
              <a href="https://github.com" target="_blank" rel="noreferrer"
                className="hover:text-primary dark:hover:text-[#3B82F6] transition-colors flex items-center gap-1">
                GitHub
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>north_east</span>
              </a>
              <button onClick={toggleTheme} className="text-outline hover:opacity-70 transition-opacity">
                <span className="material-symbols-outlined" style={{ fontSize: 20 }}>
                  {isDark ? 'light_mode' : 'dark_mode'}
                </span>
              </button>
            </div>
          </header>

          {/* Mobile header */}
          <header className="md:hidden sticky top-0 z-40 w-full bg-surface/80 dark:bg-[#0F0F0F]/80 backdrop-blur-xl flex justify-between items-center h-16 px-4">
            <div className="text-[15px] font-semibold tracking-[-0.01em] text-[#111111] dark:text-[#F8F9FF]">
              StyleVector
            </div>
            <button onClick={toggleTheme} className="text-[#2563EB] dark:text-[#3B82F6]">
              <span className="material-symbols-outlined">{isDark ? 'light_mode' : 'dark_mode'}</span>
            </button>
          </header>

          {/* Empty state */}
          <div className="flex-1 flex flex-col items-center justify-center p-8 max-w-3xl mx-auto w-full pb-52">
            <div className="w-16 h-16 rounded-2xl bg-surface-container-low dark:bg-[#1C1C1C] flex items-center justify-center mb-6 text-primary dark:text-[#3B82F6] shadow-[0_12px_32px_rgba(21,28,37,0.06)] border border-outline-variant/10">
              <span className="material-symbols-outlined" style={{ fontSize: 32, fontVariationSettings: "'FILL' 1" }}>
                layers
              </span>
            </div>
            <h2 className="text-[22px] font-semibold tracking-[-0.03em] text-on-surface dark:text-[#F8F9FF] mb-2">
              Cold-Start StyleVector
            </h2>
            <p className="text-[15px] text-on-surface-variant dark:text-[#9CA3AF] mb-10 text-center max-w-md leading-relaxed">
              Select a journalist and source publication to initialize the style steering model. Or begin with a raw text block.
            </p>

            {/* Feature chips */}
            <div className="flex flex-wrap items-center justify-center gap-3">
              {FEATURE_CHIPS.map(chip => (
                <div
                  key={chip}
                  className="px-4 py-2 bg-[#DBEAFE] dark:bg-[#1E3A5F] text-[#004AC6] dark:text-[#3B82F6] text-[13px] font-medium rounded-full flex items-center gap-2"
                >
                  <span>✦</span> {chip}
                </div>
              ))}
            </div>

            {error && (
              <div className="mt-6 px-4 py-3 bg-error-container text-error rounded-lg text-[13px] text-center max-w-md">
                {error}
              </div>
            )}
          </div>

          {/* Input bar — sticky bottom */}
          <InputBar onGenerate={handleGenerate} isGenerating={isGenerating} />
        </main>
      </div>
    </div>
  );
}
