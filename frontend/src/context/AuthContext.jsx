import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

// Helper function to check if JWT token is expired
const isTokenExpired = (token) => {
    try {
        const parts = token.split('.');
        if (parts.length !== 3) return true;
        
        // Decode the payload (second part)
        const decoded = JSON.parse(atob(parts[1]));
        const exp = decoded.exp;
        
        if (!exp) return true;
        
        // Check if token expiration time has passed (convert to milliseconds)
        const currentTime = Math.floor(Date.now() / 1000);
        return exp < currentTime;
    } catch (error) {
        console.error("Error decoding token:", error);
        return true;
    }
};

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Check for stored user/token on mount
        const initAuth = async () => {
            const storedToken = localStorage.getItem('token');

            if (storedToken) {
                // Check token expiration before making backend call
                if (isTokenExpired(storedToken)) {
                    console.warn("Token expired, logging out");
                    localStorage.removeItem('user');
                    localStorage.removeItem('token');
                    setUser(null);
                    setLoading(false);
                    return;
                }

                try {
                    // Verify token with backend
                    const response = await fetch('/api/v1/users/me', {
                        headers: {
                            'Authorization': `Bearer ${storedToken}`
                        }
                    });

                    if (response.ok) {
                        const userData = await response.json();
                        setUser(userData);
                        // Update stored user data if changed
                        localStorage.setItem('user', JSON.stringify(userData));
                    } else {
                        // Token invalid or expired
                        console.warn("Token invalid, logging out");
                        localStorage.removeItem('user');
                        localStorage.removeItem('token');
                        setUser(null);
                    }
                } catch (error) {
                    console.error("Failed to verify token", error);
                    // If network error, maybe keep user logged in or not? 
                    // Safer to force login if we can't verify.
                    localStorage.removeItem('user');
                    localStorage.removeItem('token');
                    setUser(null);
                }
            } else {
                setUser(null);
            }
            setLoading(false);
        };
        initAuth();
    }, []);

    const login = async (email, password) => {
        try {
            const response = await fetch('/api/v1/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Login failed');
            }

            const data = await response.json();
            localStorage.setItem('token', data.access_token);
            localStorage.setItem('user', JSON.stringify(data.user));
            setUser(data.user);
            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    };

    const signup = async (name, email, password) => {
        try {
            const response = await fetch('/api/v1/auth/signup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, email, password }),
            });

            if (!response.ok) {
                try {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Signup failed');
                } catch (parseError) {
                    throw new Error('Signup failed');
                }
            }

            try {
                const data = await response.json();
                // Returns { message: "...", otp_debug: "..." } (if supported)
                return { success: true, otpDebug: data.otp_debug };
            } catch (parseError) {
                throw new Error('Invalid response from server');
            }
        } catch (error) {
            return { success: false, error: error.message };
        }
    };

    const verifyOtp = async (email, otp) => {
        try {
            const response = await fetch('/api/v1/auth/verify-otp', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, otp }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Verification failed');
            }

            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    };

    const googleLogin = async (token) => {
        try {
            const response = await fetch('/api/v1/auth/google', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ token }),
            });

            if (!response.ok) {
                throw new Error('Google login failed');
            }

            const data = await response.json();
            localStorage.setItem('token', data.access_token);
            localStorage.setItem('user', JSON.stringify(data.user));
            setUser(data.user);
            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    const logout = () => {
        setUser(null);
        localStorage.removeItem('user');
        localStorage.removeItem('token');
    };

    const value = {
        user,
        login,
        signup,
        verifyOtp,
        googleLogin,
        logout,
        loading
    };

    return (
        <AuthContext.Provider value={value}>
            {!loading && children}
        </AuthContext.Provider>
    );
};
