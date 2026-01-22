import React, { useState, useEffect } from 'react';
import { Helmet } from 'react-helmet-async';
import { useAuth } from '../context/AuthContext';
import { User, Shield, Phone, Mail, LogOut, Clock, Calendar, CheckCircle, X } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';

export default function Profile() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const [activeTab, setActiveTab] = useState('history');

    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!user) {
            navigate('/login');
            return;
        }

        const fetchHistory = async () => {
            try {
                const response = await fetch('http://localhost:8000/api/v1/scan/history', {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('token')}` // Assuming token is stored in localStorage
                    }
                });

                if (response.ok) {
                    const data = await response.json();
                    // Transform backend data to frontend format
                    const formattedHistory = data.map(item => ({
                        id: item.id,
                        date: new Date(item.created_at).toISOString().split('T')[0],
                        type: 'Scan',
                        result: item.disease_detected,
                        status: 'Detected'
                    }));
                    setHistory(formattedHistory);
                } else {
                    console.error('Failed to fetch history');
                }
            } catch (error) {
                console.error('Error fetching history:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchHistory();
    }, [user, navigate]);

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    if (!user) return null;

    const userData = {
        name: user?.name || "User",
        email: user?.email,
        phone: "+1 (555) 123-4567",
        memberSince: user?.created_at
            ? new Date(user.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'short' })
            : "Jan 2026"
    };

    const [isChangePasswordOpen, setIsChangePasswordOpen] = useState(false);
    const [changePasswordForm, setChangePasswordForm] = useState({
        currentPassword: '',
        newPassword: '',
        confirmPassword: ''
    });
    const [changePasswordStatus, setChangePasswordStatus] = useState({ type: '', message: '' });

    const handleChangePassword = async (e) => {
        e.preventDefault();
        setChangePasswordStatus({ type: '', message: '' });

        if (changePasswordForm.newPassword !== changePasswordForm.confirmPassword) {
            setChangePasswordStatus({ type: 'error', message: 'New passwords do not match' });
            return;
        }

        try {
            const response = await fetch('http://localhost:8000/api/v1/users/change-password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({
                    current_password: changePasswordForm.currentPassword,
                    new_password: changePasswordForm.newPassword
                })
            });

            const data = await response.json();

            if (response.ok) {
                setChangePasswordStatus({ type: 'success', message: 'Password updated successfully' });
                setChangePasswordForm({ currentPassword: '', newPassword: '', confirmPassword: '' });
                setTimeout(() => setIsChangePasswordOpen(false), 2000);
            } else {
                setChangePasswordStatus({ type: 'error', message: data.detail || 'Failed to update password' });
            }
        } catch (error) {
            setChangePasswordStatus({ type: 'error', message: 'An error occurred. Please try again.' });
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 pt-24 pb-12">
            <Helmet>
                <title>Profile - SkinAi</title>
            </Helmet>

            <div className="max-w-4xl mx-auto px-4 sm:px-6">
                <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
                    {/* Header */}
                    <div className="bg-medical-900 h-32 relative">
                        <div className="absolute -bottom-12 left-8">
                            <div className="w-24 h-24 bg-white rounded-full p-1 shadow-lg pointer-events-none select-none">
                                <div className="w-full h-full bg-medical-100 rounded-full flex items-center justify-center text-medical-600">
                                    <User className="w-10 h-10" />
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="pt-16 pb-8 px-8">
                        <div className="flex justify-between items-start mb-6">
                            <div>
                                <h1 className="text-2xl font-bold text-gray-900">{userData.name}</h1>
                                <div className="flex items-center text-gray-500 mt-1 space-x-4 text-sm">
                                    <span className="flex items-center"><Mail className="w-4 h-4 mr-1" /> {userData.email}</span>
                                    <span className="flex items-center"><Calendar className="w-4 h-4 mr-1" /> Member since {userData.memberSince}</span>
                                </div>
                            </div>
                            <button
                                onClick={handleLogout}
                                className="flex items-center px-4 py-2 border border-red-200 text-red-600 rounded-lg hover:bg-red-50 transition"
                            >
                                <LogOut className="w-4 h-4 mr-2" /> Logout
                            </button>
                        </div>

                        <div className="border-b border-gray-200 mb-8">
                            <nav className="-mb-px flex space-x-8">
                                <button
                                    onClick={() => setActiveTab('history')}
                                    className={`${activeTab === 'history' ? 'border-medical-500 text-medical-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'} whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                                >
                                    Detection History
                                </button>
                                <button
                                    onClick={() => setActiveTab('security')}
                                    className={`${activeTab === 'security' ? 'border-medical-500 text-medical-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'} whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                                >
                                    Security & Privacy
                                </button>
                            </nav>
                        </div>

                        {activeTab === 'history' && (
                            <div className="space-y-4">
                                {loading ? (
                                    <div className="text-center py-8 text-gray-500">Loading history...</div>
                                ) : history.length > 0 ? (
                                    history.map((item) => (
                                        <div key={item.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border border-gray-100 hover:border-medical-200 transition">
                                            <div className="flex items-center">
                                                <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center border border-gray-200 mr-4">
                                                    <Clock className="w-5 h-5 text-gray-400" />
                                                </div>
                                                <div>
                                                    <p className="font-semibold text-gray-900">{item.result}</p>
                                                    <p className="text-sm text-gray-500">{item.type} • {item.date}</p>
                                                </div>
                                            </div>
                                            <span className="px-3 py-1 bg-white border border-gray-200 rounded-full text-xs font-medium text-gray-600">
                                                {item.status}
                                            </span>
                                        </div>
                                    ))
                                ) : (
                                    <div className="text-center py-8 text-gray-500">No scan history found.</div>
                                )}
                                <div className="mt-4 text-center">
                                    <Link to="/track" className="text-medical-600 font-medium hover:underline text-sm">View full analytics</Link>
                                </div>
                            </div>
                        )}

                        {activeTab === 'security' && (
                            <div className="space-y-6">
                                <div className="bg-blue-50 p-4 rounded-lg flex items-start">
                                    <Shield className="w-5 h-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
                                    <div>
                                        <h4 className="text-blue-900 font-medium text-sm">Medical Privacy Disclaimer</h4>
                                        <p className="text-blue-700 text-sm mt-1">
                                            Your data is encrypted and stored securely compliant with HIPAA guidelines. We do not share your personal health information with third parties without your explicit consent.
                                        </p>
                                    </div>
                                </div>

                                <div>
                                    <h4 className="font-medium text-gray-900 mb-2">Account Security</h4>
                                    <p className="text-sm text-gray-500 mb-4">Manage your password and authentication methods.</p>
                                    <button
                                        onClick={() => setIsChangePasswordOpen(true)}
                                        className="text-medical-600 text-sm font-medium hover:bg-medical-50 px-3 py-2 rounded-lg -ml-3 transition"
                                    >
                                        Change Password
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
                {/* Debug Info */}
                <div className="mt-8 p-4 bg-gray-100 rounded-lg overflow-auto">
                    <p className="text-xs font-mono text-gray-500 mb-2">Debug Data (Developer Only):</p>
                    <pre className="text-xs text-gray-700">{JSON.stringify(user, null, 2)}</pre>
                </div>

                {/* Change Password Modal */}
                {isChangePasswordOpen && (
                    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
                        <div className="bg-white rounded-xl shadow-xl max-w-md w-full p-6 relative">
                            <button
                                onClick={() => setIsChangePasswordOpen(false)}
                                className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
                            >
                                <X className="w-5 h-5" />
                            </button>

                            <h2 className="text-xl font-bold text-gray-900 mb-4">Change Password</h2>

                            {changePasswordStatus.message && (
                                <div className={`mb-4 p-3 rounded-lg text-sm ${changePasswordStatus.type === 'success' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
                                    }`}>
                                    {changePasswordStatus.message}
                                </div>
                            )}

                            <form onSubmit={handleChangePassword} className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Current Password</label>
                                    <input
                                        type="password"
                                        required
                                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-medical-500 focus:border-medical-500 outline-none transition"
                                        value={changePasswordForm.currentPassword}
                                        onChange={(e) => setChangePasswordForm({ ...changePasswordForm, currentPassword: e.target.value })}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">New Password</label>
                                    <input
                                        type="password"
                                        required
                                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-medical-500 focus:border-medical-500 outline-none transition"
                                        value={changePasswordForm.newPassword}
                                        onChange={(e) => setChangePasswordForm({ ...changePasswordForm, newPassword: e.target.value })}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Confirm New Password</label>
                                    <input
                                        type="password"
                                        required
                                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-medical-500 focus:border-medical-500 outline-none transition"
                                        value={changePasswordForm.confirmPassword}
                                        onChange={(e) => setChangePasswordForm({ ...changePasswordForm, confirmPassword: e.target.value })}
                                    />
                                </div>

                                <div className="flex justify-end space-x-3 pt-2">
                                    <button
                                        type="button"
                                        onClick={() => setIsChangePasswordOpen(false)}
                                        className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        type="submit"
                                        className="px-4 py-2 bg-medical-600 text-white rounded-lg hover:bg-medical-700 transition"
                                    >
                                        Update Password
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
