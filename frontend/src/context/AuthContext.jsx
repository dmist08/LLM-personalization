import { createContext, useContext, useState, useEffect } from 'react';

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

  const login = (userData) => {
    setUser(userData);
    localStorage.setItem('sv_user', JSON.stringify(userData));
  };

  const loginAsGuest = () => {
    const guest = { id: 'guest_' + Date.now(), name: 'Guest', email: null, isGuest: true };
    login(guest);
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('sv_user');
  };

  return (
    <AuthContext.Provider value={{ user, login, loginAsGuest, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
