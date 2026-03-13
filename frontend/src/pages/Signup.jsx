import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import AuthLayout from '../components/auth/AuthLayout';
import { Mail, User, Lock } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

import { useGoogleLogin } from '@react-oauth/google';

export default function Signup() {
    const [email, setEmail] = useState('');
    const [name, setName] = useState('');
    const [password, setPassword] = useState(''); // Added password state
    const navigate = useNavigate();

    const { signup, googleLogin } = useAuth();
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleGoogleSignup = useGoogleLogin({
        onSuccess: async (tokenResponse) => {
            console.log("Google Auth Success:", tokenResponse);
            setLoading(true);
            try {
                const res = await googleLogin(tokenResponse.access_token);
                console.log("Backend Google Login Response:", res);
                setLoading(false);
                if (res.success) {
                    navigate('/profile');
                } else {
                    console.error("Backend Error:", res.error);
                    setError(res.error || 'Google signup failed');
                }
            } catch (err) {
                console.error("Frontend Error during backend call:", err);
                setError("Network error or backend unreachable");
                setLoading(false);
            }
        },
        onError: (errorResponse) => {
            console.error("Google Auth Error:", errorResponse);
            setError('Google signup failed');
        }
    });

    const handleSignup = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            const res = await signup(name, email, password);
            setLoading(false);

            if (res.success) {
                navigate('/otp', { state: { email, type: 'signup' } });
            } else {
                console.warn("Signup Rejected:", res.error);
                if (res.error === "Email already registered") {
                    setError(
                        <div className="flex flex-col items-center p-2 bg-red-100 border-2 border-red-500 rounded-md">
                            <span className="font-bold text-red-700 underline text-lg">EMAIL ALREADY REGISTERED</span>
                            <Link to="/login" className="text-blue-600 font-bold underline mt-2 text-base">Please CLICK HERE to Login</Link>
                        </div>
                    );
                } else {
                    setError(res.error);
                }
            }
        } catch (err) {
            setLoading(false);
            setError(err.message || 'Failed to sign up. Please try again.');
        }
    };

    return (
        <AuthLayout
            title="Create Account"
            subtitle="Start your skin health journey today."
        >
            <form onSubmit={handleSignup} className="space-y-6">
                <div>
                    <label htmlFor="name" className="block text-sm font-medium text-gray-500 dark:text-gray-400">Full Name</label>
                    <div className="mt-1 relative rounded-md shadow-sm">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <User className="h-5 w-5 text-gray-400" />
                        </div>
                        <input
                            type="text"
                            id="name"
                            required
                            className="focus:ring-medical-500 focus:border-medical-500 block w-full pl-10 sm:text-sm bg-gray-50 dark:bg-slate-800/50 border-gray-300 dark:border-slate-700 rounded-lg py-3 text-gray-900 dark:text-white"
                            placeholder="John Doe"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                        />
                    </div>
                </div>

                <div>
                    <label htmlFor="email" className="block text-sm font-medium text-gray-500 dark:text-gray-400">Email Address</label>
                    <div className="mt-1 relative rounded-md shadow-sm">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <Mail className="h-5 w-5 text-gray-400" />
                        </div>
                        <input
                            type="email"
                            id="email"
                            required
                            className="focus:ring-medical-500 focus:border-medical-500 block w-full pl-10 sm:text-sm bg-gray-50 dark:bg-slate-800/50 border-gray-300 dark:border-slate-700 rounded-lg py-3 text-gray-900 dark:text-white"
                            placeholder="you@example.com"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                        />
                    </div>
                </div>

                <div>
                    <label htmlFor="password" className="block text-sm font-medium text-gray-500 dark:text-gray-400">Password</label>
                    <div className="mt-1 relative rounded-md shadow-sm">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <Lock className="h-5 w-5 text-gray-400" />
                        </div>
                        <input
                            type="password"
                            id="password"
                            required
                            className="focus:ring-medical-500 focus:border-medical-500 block w-full pl-10 sm:text-sm bg-gray-50 dark:bg-slate-800/50 border-gray-300 dark:border-slate-700 rounded-lg py-3 text-gray-900 dark:text-white"
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
                    className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-medical-600 hover:bg-medical-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-medical-500 transition-colors disabled:opacity-50"
                >
                    {loading ? 'Creating Account...' : 'Create Account'}
                </button>

                <div className="relative">
                    <div className="absolute inset-0 flex items-center">
                        <div className="w-full border-t border-gray-200 dark:border-slate-700" />
                    </div>
                    <div className="relative flex justify-center text-sm">
                        <span className="px-2 bg-white dark:bg-slate-900 text-gray-500 font-medium italic">Or sign up with</span>
                    </div>
                </div>

                <button
                    type="button"
                    onClick={() => handleGoogleSignup()}
                    className="w-full flex justify-center items-center py-3 px-4 border border-gray-300 dark:border-slate-700 rounded-lg shadow-sm bg-white dark:bg-slate-800/50 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-slate-700 transition-colors"
                >
                    <img className="h-5 w-5 mr-3" src="https://www.svgrepo.com/show/475656/google-color.svg" alt="Google" />
                    Google
                </button>

                <p className="text-center text-sm text-gray-500">
                    Already a member?{' '}
                    <Link to="/login" className="font-semibold text-medical-600 hover:text-medical-500">
                        Log in
                    </Link>
                </p>
            </form>
        </AuthLayout>
    );
}
