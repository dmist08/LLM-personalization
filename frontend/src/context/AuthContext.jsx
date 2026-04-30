import { createContext, useContext, useState, useEffect, useCallback } from 'react';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    try {
      const saved = localStorage.getItem('sv_user');
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  const login = useCallback((userData) => {
    setUser(userData);
    localStorage.setItem('sv_user', JSON.stringify(userData));
  }, []);

  const loginAsGuest = useCallback(() => {
    const guest = { id: 'guest_' + Date.now(), name: 'Guest', email: null, isGuest: true };
    login(guest);
  }, [login]);

  const loginWithGoogle = useCallback((credentialResponse) => {
    // Decode the JWT from Google Identity Services
    try {
      const payload = JSON.parse(atob(credentialResponse.credential.split('.')[1]));
      const googleUser = {
        id: payload.sub, // Google unique user ID — stable across sessions
        name: payload.name,
        email: payload.email,
        picture: payload.picture,
        isGuest: false,
      };
      login(googleUser);
      return googleUser;
    } catch (err) {
      console.error('Failed to decode Google credential:', err);
      return null;
    }
  }, [login]);

  const logout = useCallback(() => {
    setUser(null);
    localStorage.removeItem('sv_user');
  }, []);

  return (
    <AuthContext.Provider value={{ user, login, loginAsGuest, loginWithGoogle, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
