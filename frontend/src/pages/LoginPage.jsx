import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';

export default function LoginPage() {
  const { login, loginAsGuest } = useAuth();
  const { isDark, toggleTheme } = useTheme();
  const navigate = useNavigate();

  const handleGoogle = () => {
    // TODO: wire to real Google OAuth (Firebase / Supabase / NextAuth)
    // For now simulates a login
    login({ id: 'google_' + Date.now(), name: 'Demo User', email: 'demo@stylevector.ai' });
    navigate('/');
  };

  const handleMagicLink = () => {
    // TODO: wire to real magic link flow
    const email = prompt('Enter your email for a magic link:');
    if (email) {
      alert(`Magic link sent to ${email}! (simulated)`);
    }
  };

  const handleGuest = () => {
    loginAsGuest();
    navigate('/');
  };

  return (
    <div className={`${isDark ? 'dark' : ''}`}>
      <div className="bg-background dark:bg-[#0F0F0F] min-h-screen flex flex-col justify-center items-center text-on-surface dark:text-[#F8F9FF] antialiased p-4 transition-colors duration-300">

        {/* Login Card */}
        <div className="bg-surface-container-lowest dark:bg-[#1C1C1C] rounded-lg p-10 w-full max-w-[400px] shadow-[0_12px_32px_rgba(21,28,37,0.06)] dark:shadow-[0_12px_32px_rgba(0,0,0,0.40)] relative flex flex-col">

          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="absolute top-4 right-4 text-outline hover:opacity-70 transition-opacity"
            aria-label="Toggle theme"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>
              {isDark ? 'light_mode' : 'dark_mode'}
            </span>
          </button>

          {/* Header */}
          <div className="text-center mb-8 mt-2">
            <h1 className="text-[20px] font-semibold text-[#111111] dark:text-[#F8F9FF] tracking-[-0.01em] mb-2">
              Cold-Start StyleVector
            </h1>
            <p className="text-[13px] text-outline dark:text-[#9CA3AF]">
              Personalized headline generation for journalists
            </p>
          </div>

          {/* Tonal Divider */}
          <div className="w-full h-px bg-surface-container-high dark:bg-[#2a2a2a] mb-8" />

          {/* Auth Actions */}
          <div className="flex flex-col gap-4 mb-6">
            <button
              onClick={handleGoogle}
              className="w-full h-[44px] bg-surface-container-lowest dark:bg-[#1C1C1C] ghost-border rounded-lg flex items-center justify-center gap-3 hover:bg-surface-container-low dark:hover:bg-[#252525] transition-colors group"
            >
              <img
                alt="Google"
                className="w-5 h-5 group-hover:scale-105 transition-transform"
                src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg"
              />
              <span className="text-[14px] font-medium text-on-surface dark:text-[#F8F9FF]">
                Continue with Google
              </span>
            </button>

            <button
              onClick={handleMagicLink}
              className="w-full h-[44px] bg-surface-container-lowest dark:bg-[#1C1C1C] ghost-border rounded-lg flex items-center justify-center gap-3 hover:bg-surface-container-low dark:hover:bg-[#252525] transition-colors group"
            >
              <span className="material-symbols-outlined text-on-surface dark:text-[#F8F9FF] group-hover:scale-105 transition-transform" style={{ fontSize: 20 }}>
                mail
              </span>
              <span className="text-[14px] font-medium text-on-surface dark:text-[#F8F9FF]">
                Continue with Magic Link
              </span>
            </button>
          </div>

          {/* Guest Link */}
          <div className="text-center mt-2">
            <button
              onClick={handleGuest}
              className="text-[13px] text-primary dark:text-[#3B82F6] hover:opacity-80 transition-opacity"
            >
              Continue as Guest
            </button>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-[12px] text-outline tracking-wider uppercase">
          DA-IICT · DEEP LEARNING IT549 · 2026
        </div>
      </div>
    </div>
  );
}
