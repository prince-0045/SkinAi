import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import AuthLayout from '../components/auth/AuthLayout';
import { useAuth } from '../context/AuthContext';
import { Mail, Lock, ArrowRight } from 'lucide-react';
import { useGoogleLogin } from '@react-oauth/google';

export default function Login() {
    const [email, setEmail] = useState('');
    const navigate = useNavigate();

    const { login, googleLogin } = useAuth();
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleGoogleLogin = useGoogleLogin({
        onSuccess: async (tokenResponse) => {
            console.log("Google Login Success:", tokenResponse);
            setLoading(true);
            try {
                const res = await googleLogin(tokenResponse.access_token);
                console.log("Backend Google Login Response:", res);
                setLoading(false);
                if (res.success) {
                    navigate('/track');
                } else {
                    console.error("Backend Error:", res.error);
                    setError(res.error || 'Google login failed');
                }
            } catch (err) {
                console.error("Frontend Error during backend call:", err);
                setError("Network error or backend unreachable");
                setLoading(false);
            }
        },
        onError: (errorResponse) => {
            console.error("Google Login Error:", errorResponse);
            setError('Google login failed');
        }
    });

    const handleLogin = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        const res = await login(email, password);
        setLoading(false);

        if (res.success) {
            navigate('/track'); // Redirect to dashboard
        } else {
            if (res.error === "Email not verified") {
                // Redirect to OTP if not verified
                navigate('/otp', { state: { email, type: 'login' } });
            } else {
                setError(res.error);
            }
        }
    };

    return (
        <AuthLayout
            title="Welcome Back"
            subtitle="Log in to access your healing dashboard."
        >
            <form onSubmit={handleLogin} className="space-y-6">
                <div>
                    <label htmlFor="email" className="block text-sm font-medium text-gray-700">Email Address</label>
                    <div className="mt-1 relative rounded-md shadow-sm">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <Mail className="h-5 w-5 text-gray-400" />
                        </div>
                        <input
                            type="email"
                            name="email"
                            id="email"
                            required
                            className="focus:ring-medical-500 focus:border-medical-500 block w-full pl-10 sm:text-sm border-gray-300 rounded-lg py-3"
                            placeholder="you@example.com"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                        />
                    </div>
                </div>

                <div>
                    <label htmlFor="password" className="block text-sm font-medium text-gray-700">Password</label>
                    <div className="mt-1 relative rounded-md shadow-sm">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <Lock className="h-5 w-5 text-gray-400" />
                        </div>
                        <input
                            type="password"
                            name="password"
                            id="password"
                            required
                            className="focus:ring-medical-500 focus:border-medical-500 block w-full pl-10 sm:text-sm border-gray-300 rounded-lg py-3"
                            placeholder="••••••••"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                        />
                    </div>
                </div>

                {error && (
                    <div className="text-red-500 text-sm text-center bg-red-50 p-2 rounded">
                        {error}
                    </div>
                )}

                <button
                    type="submit"
                    disabled={loading}
                    className="w-full flex justify-center items-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-medical-600 hover:bg-medical-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-medical-500 transition-colors disabled:opacity-50"
                >
                    {loading ? 'Logging in...' : 'Login'} <ArrowRight className="ml-2 w-4 h-4" />
                </button>

                <div className="relative">
                    <div className="absolute inset-0 flex items-center">
                        <div className="w-full border-t border-gray-200" />
                    </div>
                    <div className="relative flex justify-center text-sm">
                        <span className="px-2 bg-white text-gray-500">Or continue with</span>
                    </div>
                </div>

                <button
                    type="button"
                    onClick={() => handleGoogleLogin()}
                    className="w-full flex justify-center items-center py-3 px-4 border border-gray-300 rounded-lg shadow-sm bg-white text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
                >
                    <img className="h-5 w-5 mr-3" src="https://www.svgrepo.com/show/475656/google-color.svg" alt="Google" />
                    Sign in with Google
                </button>

                <p className="text-center text-sm text-gray-500">
                    Don't have an account?{' '}
                    <Link to="/signup" className="font-semibold text-medical-600 hover:text-medical-500">
                        Sign up
                    </Link>
                </p>
            </form>
        </AuthLayout>
    );
}
