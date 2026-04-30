import { useNavigate, useParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { useState, useEffect } from 'react';
import { getChatHistory, deleteChatSession } from '../services/api';

const TRANSITION = 'all 0.28s cubic-bezier(0.4,0,0.2,1)';
const EXPANDED_W = 260;
const COLLAPSED_W = 64;

export default function Sidebar({ onNewChat, onCollapse }) {
  const { user, logout } = useAuth();
  const { isDark } = useTheme();
  const navigate = useNavigate();
  const { sessionId: activeSessionId } = useParams();
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    if (!user) return;
    getChatHistory(user.id)
      .then(({ sessions }) => setSessions(sessions || []))
      .catch(() => setSessions([]))
      .finally(() => setLoading(false));
  }, [user, activeSessionId]);

  useEffect(() => { onCollapse?.(collapsed); }, [collapsed]);

  const toggle = () => setCollapsed(c => !c);

  const handleDelete = async (e, id) => {
    e.stopPropagation();
    await deleteChatSession(id);
    setSessions(prev => prev.filter(s => s.id !== id));
    if (activeSessionId === id) navigate('/');
  };

  const navItems = [
    { icon: 'home', label: 'Home', path: '/' },
    { icon: 'inventory_2', label: 'Archive', path: '/archive' },
    { icon: 'settings', label: 'Settings', path: '/settings' },
    { icon: 'help', label: 'Support', path: '/support' },
  ];

  /* Shared style helpers */
  const labelStyle = {
    opacity: collapsed ? 0 : 1,
    maxWidth: collapsed ? 0 : 160,
    overflow: 'hidden',
    whiteSpace: 'nowrap',
    transition: TRANSITION,
    display: 'block',
  };

  const sidebarStyle = {
    width: collapsed ? COLLAPSED_W : EXPANDED_W,
    minWidth: collapsed ? COLLAPSED_W : EXPANDED_W,
    transition: TRANSITION,
    overflow: 'hidden',
  };

  return (
    <nav
      style={sidebarStyle}
      className="hidden md:flex flex-col h-screen py-6 fixed left-0 top-0 z-50
                 bg-[#EEF4FF] dark:bg-[#171717] transition-colors duration-300"
    >
      {/* ── Brand ─────────────────────────────────────── */}
      <div className="flex items-center gap-3 mb-8 px-4">
        {/* Avatar doubles as toggle */}
        <button
          onClick={toggle}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="w-8 h-8 rounded-full bg-primary-container dark:bg-[#1E3A5F]
                     text-white flex items-center justify-center font-semibold
                     text-xs flex-shrink-0 hover:opacity-80 active:scale-95 transition-all overflow-hidden"
        >
          {user?.picture ? (
            <img src={user.picture} alt={user.name} className="w-full h-full object-cover" referrerPolicy="no-referrer" />
          ) : (
            user?.name?.substring(0, 2).toUpperCase() || 'SV'
          )}
        </button>

        <div style={labelStyle} className="flex flex-col">
          <span className="text-[15px] font-semibold tracking-[-0.01em] text-[#111111] dark:text-[#F8F9FF] truncate">
            StyleVector
          </span><br></br>
          <span className="text-[11px] font-medium tracking-[0.08em] text-outline uppercase truncate">
            Editorial Intelligence
          </span>
        </div>
      </div>

      {/* ── New Chat ──────────────────────────────────── */}
      <div className="px-3 mb-6">
        <button
          onClick={onNewChat}
          title="New Chat"
          className="w-full h-9 bg-gradient-to-br from-[#004AC6] to-[#2563EB] text-white
                     rounded-[6px] flex items-center justify-center gap-2
                     hover:brightness-110 active:scale-95 transition-all
                     shadow-[0_4px_12px_rgba(37,99,235,0.2)] text-[13px] font-medium overflow-hidden"
        >
          <span className="material-symbols-outlined flex-shrink-0" style={{ fontSize: 16 }}>add</span>
          <span style={labelStyle}>New Chat</span>
        </button>
      </div>

      {/* ── Nav Links ─────────────────────────────────── */}
      <div className="flex flex-col gap-1 px-2">
        {navItems.map(item => {
          const active = location.pathname === item.path;
          return (
            <button
              key={item.label}
              onClick={() => navigate(item.path)}
              title={item.label}
              className={`px-2 py-1.5 flex items-center gap-3 rounded-md text-[13px] transition-colors overflow-hidden ${active
                ? 'text-[#2563EB] dark:text-[#3B82F6] font-semibold bg-[#DBEAFE] dark:bg-[#1E3A5F]'
                : 'text-slate-500 dark:text-slate-400 hover:bg-[#DBEAFE] dark:hover:bg-[#1E3A5F]'
                }`}
            >
              <span className="material-symbols-outlined flex-shrink-0" style={{ fontSize: 20 }}>
                {item.icon}
              </span>
              <span style={labelStyle}>{item.label}</span>
            </button>
          );
        })}
      </div>

      {/* ── Recent Chats ──────────────────────────────── */}
      <div
        className="flex-grow mt-8 overflow-y-auto no-scrollbar"
        style={{
          opacity: collapsed ? 0 : 1,
          transition: TRANSITION,
          pointerEvents: collapsed ? 'none' : 'auto',
        }}
      >
        <div className="px-4 text-[11px] font-medium tracking-[0.08em] text-outline uppercase mb-3">
          Recent
        </div>

        {loading ? (
          <div className="space-y-2 px-2">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-12 bg-surface-container dark:bg-[#1C1C1C] rounded-md animate-shimmer" />
            ))}
          </div>
        ) : sessions.length === 0 ? (
          <div className="px-4 py-1.5 text-[13px] text-outline-variant italic">
            No chats yet
          </div>
        ) : (
          <div className="flex flex-col gap-1 px-2">
            {sessions.map(session => (
              <div
                key={session.id}
                onClick={() => navigate(`/chat/${session.id}`)}
                className={`group relative bg-surface-container-lowest dark:bg-[#1C1C1C] p-3
                            rounded-[8px] cursor-pointer hover:bg-[#DBEAFE] dark:hover:bg-[#1E3A5F]
                            transition-colors ${activeSessionId === session.id ? 'border-l-2 border-primary' : ''
                  }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[12px] font-semibold text-on-surface dark:text-[#F8F9FF]">
                    {session.author_name}
                  </span>
                  <span className="text-[9px] bg-surface-container-high dark:bg-[#2a2a2a] px-1.5 py-0.5
                                   rounded text-on-surface-variant dark:text-[#9CA3AF] font-bold uppercase">
                    {session.publication_code}
                  </span>
                </div>
                <p className="text-[11px] text-on-surface-variant dark:text-[#9CA3AF] truncate pr-4">
                  {session.preview}
                </p>
                <button
                  onClick={(e) => handleDelete(e, session.id)}
                  className="absolute right-2 top-2 opacity-0 group-hover:opacity-100
                             text-outline hover:text-error transition-all"
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 14 }}>delete</span>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Footer / Logout ───────────────────────────── */}
      <div className="mt-auto pt-4 border-t border-outline-variant/10 px-2">
        {/* Email */}
        <div
          className="px-2 py-1 text-[12px] text-outline truncate mb-2"
          style={{
            opacity: collapsed ? 0 : 1,
            maxHeight: collapsed ? 0 : 40,
            overflow: 'hidden',
            transition: TRANSITION,
          }}
        >
          {user?.email || 'Guest user'}
        </div>

        {/* Logout */}
        <button
          onClick={logout}
          title="Logout"
          className="w-full px-2 py-1.5 flex items-center gap-3 rounded-md text-[13px]
                     text-slate-500 dark:text-slate-400 hover:bg-[#DBEAFE]
                     dark:hover:bg-[#1E3A5F] transition-colors overflow-hidden"
        >
          <span className="material-symbols-outlined flex-shrink-0" style={{ fontSize: 20 }}>logout</span>
          <span style={labelStyle}>Logout</span>
        </button>

        {/* Collapse toggle */}
        <button
          onClick={toggle}
          title={collapsed ? 'Expand' : 'Collapse'}
          className="w-full mt-1 px-2 py-1.5 flex items-center gap-3 rounded-md text-[13px]
                     text-slate-400 hover:bg-[#DBEAFE] dark:hover:bg-[#1E3A5F]
                     transition-colors overflow-hidden"
        >
          <span
            className="material-symbols-outlined flex-shrink-0"
            style={{
              fontSize: 20,
              transform: collapsed ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: TRANSITION,
            }}
          >
            chevron_left
          </span>
          <span style={labelStyle}>Collapse</span>
        </button>
      </div>
    </nav>
  );
}
