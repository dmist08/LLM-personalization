import { useParams, useLocation, useNavigate } from 'react-router-dom';
import { useState, useEffect, useRef } from 'react';
import Sidebar from '../components/Sidebar';
import InputBar from '../components/InputBar';
import HeadlineCard from '../components/HeadlineCard';
import MetricsChart from '../components/MetricsChart';
import { useTheme } from '../context/ThemeContext';
import { useAuth } from '../context/AuthContext';
import { generateHeadlines, getChatSession } from '../services/api';
import { computeMetrics } from '../utils/metrics';

const RESULT_TYPES = ['no_personalization', 'rag_bm25', 'stylevector', 'cold_start_sv'];

export default function ChatPage() {
  const { sessionId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const { isDark, toggleTheme } = useTheme();
  const { user } = useAuth();

  const [messages, setMessages] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [copiedId, setCopiedId] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const bottomRef = useRef(null);

  const sidebarW = sidebarCollapsed ? 64 : 260;
  const hasMessages = messages.length > 0;

  // Load session from route state (fresh generation) or from API (history visit)
  useEffect(() => {
    if (location.state?.results) {
      const { results, payload } = location.state;
      setMessages([{
        id: results.session_id,
        payload,
        results: results.results,
        isLoading: false,
      }]);
      window.history.replaceState({}, '');
    } else if (sessionId) {
      getChatSession(sessionId)
        .then(session => {
          setMessages(session.messages || []);
        })
        .catch(() => navigate('/'));
    }
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleGenerate = async (payload) => {
    setIsGenerating(true);
    setError(null);
    const tempId = 'pending_' + Date.now();
    setMessages(prev => [...prev, {
      id: tempId,
      payload,
      results: null,
      isLoading: true,
    }]);

    try {
      const data = await generateHeadlines({ ...payload, sessionId });
      console.log('[ChatPage] generateHeadlines raw response:', data);
      console.log('[ChatPage] data.results:', data?.results);
      setMessages(prev => {
        const updated = prev.map(m =>
          m.id === tempId
            ? { id: data.session_id, payload, results: data.results, isLoading: false }
            : m
        );
        console.log('[ChatPage] updated messages:', updated);
        return updated;
      });
    } catch (err) {
      setError(err.response?.data?.error || 'Generation failed. Please try again.');
      setMessages(prev => prev.filter(m => m.id !== tempId));
    } finally {
      setIsGenerating(false);
    }
  };

  const handleCopy = (text, key) => {
    navigator.clipboard.writeText(text);
    setCopiedId(key);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className={`${isDark ? 'dark' : ''}`}>
      <div className="bg-surface dark:bg-[#0F0F0F] text-on-surface dark:text-[#F8F9FF] h-screen flex overflow-hidden transition-colors duration-300">

        <Sidebar onNewChat={() => navigate('/')} onCollapse={setSidebarCollapsed} />

        {/* Main content — shifts with sidebar */}
        <main
          style={{
            marginLeft: sidebarW,
            transition: 'margin-left 0.25s cubic-bezier(0.4,0,0.2,1)',
          }}
          className="flex-1 h-screen flex flex-col bg-surface dark:bg-[#0F0F0F] overflow-hidden"
        >

          {/* Top header */}
          <header className="h-16 items-center justify-between px-8 bg-surface/80 dark:bg-[#0F0F0F]/80 backdrop-blur-md sticky top-0 z-30 hidden md:flex border-b border-outline-variant/10">
            <h1 className="text-[17px] font-medium tracking-[-0.02em] text-on-surface dark:text-[#F8F9FF]">
              Cold-Start StyleVector
            </h1>
            <div className="flex items-center gap-4">
              <button onClick={toggleTheme} className="text-outline hover:opacity-70 transition-opacity">
                <span className="material-symbols-outlined" style={{ fontSize: 20 }}>
                  {isDark ? 'light_mode' : 'dark_mode'}
                </span>
              </button>
            </div>
          </header>

          {/* ── No-message state: center the InputBar ── */}
          {!hasMessages ? (
            <div className="flex-1 flex flex-col items-center justify-center px-4">
              {/* Hero text */}
              <div className="mb-10 text-center select-none">
                <span
                  className="material-symbols-outlined text-primary dark:text-[#3B82F6] mb-4 block"
                  style={{ fontSize: 48 }}
                >
                  layers
                </span>
                <h2 className="text-[26px] font-semibold tracking-[-0.03em] text-on-surface dark:text-[#F8F9FF] mb-2">
                  Cold-Start StyleVector
                </h2>
                <p className="text-[14px] text-outline max-w-sm leading-relaxed">
                  Select a journalist and source publication to initialize the style
                  steering model. Or begin with a raw text block.
                </p>
                {/* Feature pills */}
                <div className="flex items-center justify-center gap-2 mt-6 flex-wrap">
                  {['✦ Activation Steering', '✦ Cold-Start Clustering', '✦ 42 Indian Journalists'].map(t => (
                    <span
                      key={t}
                      className="px-3 py-1 rounded-full border border-outline-variant/30 text-[12px] text-outline-variant dark:text-slate-400 bg-surface-container/50 dark:bg-[#1C1C1C]/50"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </div>

              {/* Floating InputBar centered */}
              <div className="w-full" style={{ maxWidth: 870 }}>
                <InputBar onGenerate={handleGenerate} isGenerating={isGenerating} />
              </div>
            </div>
          ) : (
            /* ── Has messages: scroll area + pinned InputBar ── */
            <>
              <div className="flex-1 overflow-y-auto px-4 md:px-8 py-8 pb-8 no-scrollbar max-w-[900px] mx-auto w-full">

                {messages.map((msg) => (
                  <div key={msg.id} className="mb-12">
                    {/* User prompt bubble */}
                    <div className="flex flex-col items-end mb-8">
                      <div className="flex items-center gap-2 mb-2 mr-1">
                        <span className="text-[10px] font-medium text-on-surface-variant bg-surface-container-high dark:bg-[#2a2a2a] px-2 py-0.5 rounded-full">
                          {msg.payload.authorName} · {msg.payload.publication}
                        </span>
                      </div>
                      <div className="bg-primary-container dark:bg-[#1E3A5F] text-white px-5 py-3.5 rounded-2xl rounded-tr-sm max-w-[85%] md:max-w-[70%] shadow-[0_4px_12px_rgba(37,99,235,0.15)] text-[14px] leading-relaxed">
                        {msg.payload.sourceText.length > 250
                          ? msg.payload.sourceText.substring(0, 250) + '...'
                          : msg.payload.sourceText}
                      </div>
                    </div>

                    {/* Loading / Results */}
                    {msg.isLoading ? (
                      <div>
                        <div className="flex items-center gap-3 mb-6 ml-2">
                          <span className="material-symbols-outlined text-primary dark:text-[#3B82F6] animate-spin-slow" style={{ fontSize: 18 }}>
                            progress_activity
                          </span>
                          <span className="text-[11px] font-medium uppercase tracking-[0.08em] text-primary dark:text-[#3B82F6]">
                            Generating Headlines
                          </span>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {RESULT_TYPES.map(type => (
                            <HeadlineCard key={type} type={type} isLoading={true} />
                          ))}
                        </div>
                      </div>
                    ) : msg.results ? (
                      <div>
                        <div className="flex items-center gap-2 mb-6 ml-2">
                          <span className="w-2 h-2 rounded-full bg-[#10B981] shadow-[0_0_8px_rgba(16,185,129,0.4)]" />
                          <span className="text-[11px] font-medium uppercase tracking-[0.08em] text-[#10B981]">
                            Generation Complete
                          </span>
                        </div>

                        {/* Ground truth reference */}
                        {msg.payload?.groundTruth && (
                          <div className="mb-4 px-4 py-3 rounded-lg border border-dashed border-outline-variant/30 dark:border-[#2a2a2a] bg-surface-container/30 dark:bg-[#161616]">
                            <div className="text-[10px] font-bold tracking-[0.08em] uppercase text-outline mb-1">
                              GROUND TRUTH
                            </div>
                            <p className="text-[14px] font-medium text-on-surface dark:text-[#F8F9FF]">
                              {msg.payload.groundTruth}
                            </p>
                          </div>
                        )}

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {RESULT_TYPES.map(type => {
                            const result = msg.results[type];
                            const copyKey = `${msg.id}-${type}`;
                            // Compute metrics if ground truth provided
                            const metrics = msg.payload?.groundTruth
                              ? computeMetrics(result?.headline, msg.payload.groundTruth)
                              : null;
                            return (
                              <HeadlineCard
                                key={type}
                                type={type}
                                headline={result?.headline || 'Unavailable'}
                                rougeL={metrics?.rougeL}
                                latencyMs={result?.latency_ms}
                                isLoading={false}
                                isCopied={copiedId === copyKey}
                                onCopy={(text) => handleCopy(text, copyKey)}
                              />
                            );
                          })}
                        </div>

                        {/* Metrics comparison chart */}
                        {msg.payload?.groundTruth && (() => {
                          const metricsMap = {};
                          RESULT_TYPES.forEach(type => {
                            const result = msg.results[type];
                            if (result?.headline) {
                              metricsMap[type] = computeMetrics(result.headline, msg.payload.groundTruth);
                            }
                          });
                          return <MetricsChart metricsMap={metricsMap} />;
                        })()}
                      </div>
                    ) : null}
                  </div>
                ))}

                {error && (
                  <div className="px-4 py-3 bg-error-container text-error rounded-lg text-[13px] mb-6">
                    {error}
                  </div>
                )}

                <div ref={bottomRef} />
              </div>

              {/* Pinned InputBar — flex-shrink-0, no overlap */}
              <div className="flex-shrink-0 w-full mx-auto" style={{ maxWidth: 870, padding: '0 16px' }}>
                <InputBar onGenerate={handleGenerate} isGenerating={isGenerating} />
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
