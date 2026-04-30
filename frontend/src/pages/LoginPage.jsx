import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { useEffect, useRef, useState } from 'react';

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;

export default function LoginPage() {
  const { loginWithGoogle, loginAsGuest } = useAuth();
  const { isDark, toggleTheme } = useTheme();
  const navigate = useNavigate();
  const googleBtnRef = useRef(null);
  const [gsiLoaded, setGsiLoaded] = useState(false);

  // Load Google Identity Services script
  useEffect(() => {
    if (!GOOGLE_CLIENT_ID) return;
    if (window.google?.accounts) {
      setGsiLoaded(true);
      return;
    }
    const script = document.createElement('script');
    script.src = 'https://accounts.google.com/gsi/client';
    script.async = true;
    script.defer = true;
    script.onload = () => setGsiLoaded(true);
    document.head.appendChild(script);
    return () => { /* script stays loaded */ };
  }, []);

  // Initialize Google button once script is loaded
  useEffect(() => {
    if (!gsiLoaded || !GOOGLE_CLIENT_ID || !window.google?.accounts) return;

    window.google.accounts.id.initialize({
      client_id: GOOGLE_CLIENT_ID,
      callback: (credentialResponse) => {
        const user = loginWithGoogle(credentialResponse);
        if (user) navigate('/');
      },
    });

    // Render the Google button inside our container
    window.google.accounts.id.renderButton(googleBtnRef.current, {
      theme: isDark ? 'filled_black' : 'outline',
      size: 'large',
      width: 320,
      text: 'continue_with',
      shape: 'rectangular',
      logo_alignment: 'center',
    });
  }, [gsiLoaded, isDark]);

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
          <div className="flex flex-col gap-4 mb-6 items-center">
            {/* Google Sign-In — rendered by Google's script */}
            {GOOGLE_CLIENT_ID ? (
              <div ref={googleBtnRef} className="flex justify-center" />
            ) : (
              <p className="text-[12px] text-outline italic text-center">
                Google sign-in not configured — set VITE_GOOGLE_CLIENT_ID
              </p>
            )}
          </div>

          {/* OR Divider */}
          <div className="flex items-center gap-3 mb-6">
            <div className="flex-1 h-px bg-surface-container-high dark:bg-[#2a2a2a]" />
            <span className="text-[11px] text-outline uppercase tracking-widest">or</span>
            <div className="flex-1 h-px bg-surface-container-high dark:bg-[#2a2a2a]" />
          </div>

          {/* Guest Link */}
          <div className="text-center">
            <button
              onClick={handleGuest}
              className="w-full h-[44px] bg-surface-container-lowest dark:bg-[#1C1C1C] ghost-border rounded-lg flex items-center justify-center gap-3 hover:bg-surface-container-low dark:hover:bg-[#252525] transition-colors"
            >
              <span className="material-symbols-outlined text-on-surface dark:text-[#F8F9FF]" style={{ fontSize: 20 }}>
                person
              </span>
              <span className="text-[14px] font-medium text-on-surface dark:text-[#F8F9FF]">
                Continue as Guest
              </span>
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
