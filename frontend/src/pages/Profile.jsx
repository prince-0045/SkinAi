import React, { useState, useEffect } from 'react';
import { Helmet } from 'react-helmet-async';
import { useAuth } from '../context/AuthContext';
import { User, Shield, Phone, Mail, LogOut, Clock, Calendar, CheckCircle, X, Download, Eye, EyeOff } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { downloadMedicalReport } from '../utils/pdfGenerator';

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

        const API_BASE_URL = import.meta.env.VITE_API_URL || '';
        const fetchHistory = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/api/v1/scan/history`, {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('token')}` // Assuming token is stored in localStorage
                    }
                });

                if (response.ok) {
                    const data = await response.json();
                    // Keep full data for PDF generation
                    const formattedHistory = data.map(item => ({
                        ...item,
                        date: new Date(item.created_at).toISOString().split('T')[0],
                        type: 'Scan',
                        result: item.disease_detected,
                        status: item.disease_detected === 'Unknown' ? 'Not Detected' : 'Detected'
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
    const [showPasswords, setShowPasswords] = useState(false);

    const handleChangePassword = async (e) => {
        e.preventDefault();
        setChangePasswordStatus({ type: '', message: '' });

        if (changePasswordForm.newPassword !== changePasswordForm.confirmPassword) {
            setChangePasswordStatus({ type: 'error', message: 'New passwords do not match' });
            return;
        }

        try {
            const API_BASE_URL = import.meta.env.VITE_API_URL || '';
            const response = await fetch(`${API_BASE_URL}/api/v1/users/change-password`, {
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
        <div className="bg-base min-h-screen transition-colors duration-300 pt-24 pb-12">
            <Helmet>
                <title>Profile - DERMAURA</title>
            </Helmet>

            <div className="max-w-4xl mx-auto px-4 sm:px-6 page-enter">
                <div className="profile-page-wrapper">
                    {/* Header Banner with Integrated Info */}
                    <div className="profile-banner flex flex-col sm:flex-row items-center sm:items-end justify-between px-6 sm:px-10 pb-8 pt-10 gap-6">
                        <div className="flex flex-col sm:flex-row items-center sm:items-end gap-6 w-full">
                            <div className="profile-avatar shadow-2xl flex-shrink-0">
                                <User className="w-10 h-10" />
                            </div>
                            <div className="text-center sm:text-left pb-1">
                                <h1 className="profile-name text-2xl sm:text-4xl font-extrabold text-white tracking-tight" style={{ fontFamily: 'var(--font-display)' }}>
                                    {userData.name}
                                </h1>
                                <div className="mt-3 flex flex-wrap justify-center sm:justify-start gap-x-5 gap-y-2">
                                    <span className="profile-email flex items-center gap-2 text-[var(--white-muted)] text-sm font-medium">
                                        <Mail className="w-4 h-4 text-[var(--blue-bright)]" /> {userData.email}
                                    </span>
                                    <span className="profile-member-since flex items-center gap-2 text-[var(--white-muted)] text-sm font-medium">
                                        <Calendar className="w-4 h-4 text-[var(--blue-bright)]" /> Member since {userData.memberSince}
                                    </span>
                                </div>
                            </div>
                        </div>
                        <div className="pb-1 sm:pb-2 flex-shrink-0">
                            <button
                                onClick={handleLogout}
                                className="logout-btn-profile"
                            >
                                <LogOut className="w-4 h-4" /> Logout
                            </button>
                        </div>
                    </div>

                    <div className="profile-body px-4 sm:px-8 pb-8">

                        <div className="profile-tabs flex mt-2">
                            <button
                                onClick={() => setActiveTab('history')}
                                className={`profile-tab ${activeTab === 'history' ? 'active' : ''}`}
                            >
                                Detection History
                            </button>
                            <button
                                onClick={() => setActiveTab('security')}
                                className={`profile-tab ${activeTab === 'security' ? 'active' : ''}`}
                            >
                                Security & Privacy
                            </button>
                        </div>

                        {activeTab === 'history' && (
                            <div className="history-list-section">
                                {loading ? (
                                    <div className="text-center py-8 text-[var(--white-faint)]">Loading history...</div>
                                ) : history.length > 0 ? (
                                    history.map((item) => (
                                        <div key={item.id} className="history-item">
                                            <div className="flex items-center gap-3">
                                                <div className="history-clock-icon">
                                                    <Clock className="w-5 h-5" />
                                                </div>
                                                <div>
                                                    <p className="history-item-title">{item.result}</p>
                                                    <p className="history-item-subtitle">{item.type} • {item.date}</p>
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-3">
                                                <button
                                                    onClick={() => downloadMedicalReport(item, user)}
                                                    className="history-download-btn"
                                                    title="Download Report"
                                                >
                                                    <Download className="w-5 h-5" />
                                                </button>
                                                <span className={item.status === 'Detected' ? 'history-badge-detected' : 'history-badge-not-detected'}>
                                                    {item.status}
                                                </span>
                                            </div>
                                        </div>
                                    ))
                                ) : (
                                    <div className="text-center py-8 text-[var(--white-faint)]">No scan history found.</div>
                                )}
                                <div className="mt-4 text-center">
                                    <Link to="/track" className="view-analytics-link">View full analytics</Link>
                                </div>
                            </div>
                        )}

                        {activeTab === 'security' && (
                            <div className="history-list-section pt-4">
                                <div className="bg-[#1a3a6e]/20 p-4 rounded-xl flex items-start border border-[var(--border-subtle)]">
                                    <Shield className="w-5 h-5 text-[var(--blue-bright)] mt-0.5 mr-3 flex-shrink-0" />
                                    <div>
                                        <h4 className="text-[var(--blue-bright)] font-medium text-sm" style={{ fontFamily: 'var(--font-display)' }}>Medical Privacy Disclaimer</h4>
                                        <p className="text-[var(--white-muted)] text-sm mt-1">
                                            Your data is encrypted and stored securely compliant with HIPAA guidelines. We do not share your personal health information with third parties without your explicit consent.
                                        </p>
                                    </div>
                                </div>

                                <div className="mt-6 panel p-5 card">
                                    <h4 className="font-medium text-white mb-2" style={{ fontFamily: 'var(--font-display)' }}>Account Security</h4>
                                    <p className="text-sm text-[var(--white-muted)] mb-4">Manage your password and authentication methods.</p>
                                    <button
                                        onClick={() => setIsChangePasswordOpen(true)}
                                        className="btn-secondary text-sm"
                                    >
                                        Change Password
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Change Password Modal */}
                {isChangePasswordOpen && (
                    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 z-50">
                        <div className="card max-w-md w-full p-8 relative border border-[var(--border-subtle)] shadow-2xl">
                            <button
                                onClick={() => setIsChangePasswordOpen(false)}
                                className="absolute top-4 right-4 text-[var(--white-muted)] hover:text-white transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </button>

                            <h2 className="text-2xl font-bold text-white mb-6" style={{ fontFamily: 'var(--font-display)' }}>Change Password</h2>

                            {changePasswordStatus.message && (
                                <div className={`mb-6 p-4 rounded-xl text-sm border font-medium ${changePasswordStatus.type === 'success'
                                        ? 'bg-[var(--blue-dim)] border-[var(--blue-bright)] text-[var(--blue-bright)]'
                                        : 'bg-red-500/10 border-red-500/30 text-red-400'
                                    }`}>
                                    {changePasswordStatus.message}
                                </div>
                            )}

                            <form onSubmit={handleChangePassword} className="space-y-5">
                                <div>
                                    <label className="block text-sm font-medium text-[var(--white-muted)] mb-2">Current Password</label>
                                    <div className="relative">
                                        <input
                                            type={showPasswords ? "text" : "password"}
                                            required
                                            className="input-field w-full pr-10"
                                            value={changePasswordForm.currentPassword}
                                            onChange={(e) => setChangePasswordForm({ ...changePasswordForm, currentPassword: e.target.value })}
                                        />
                                        <button
                                            type="button"
                                            onClick={() => setShowPasswords(!showPasswords)}
                                            className="absolute inset-y-0 right-0 pr-3 flex items-center text-[var(--white-faint)] hover:text-white transition"
                                        >
                                            {showPasswords ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                        </button>
                                    </div>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-[var(--white-muted)] mb-2">New Password</label>
                                    <div className="relative">
                                        <input
                                            type={showPasswords ? "text" : "password"}
                                            required
                                            className="input-field w-full"
                                            value={changePasswordForm.newPassword}
                                            onChange={(e) => setChangePasswordForm({ ...changePasswordForm, newPassword: e.target.value })}
                                        />
                                    </div>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-[var(--white-muted)] mb-2">Confirm New Password</label>
                                    <div className="relative">
                                        <input
                                            type={showPasswords ? "text" : "password"}
                                            required
                                            className="input-field w-full"
                                            value={changePasswordForm.confirmPassword}
                                            onChange={(e) => setChangePasswordForm({ ...changePasswordForm, confirmPassword: e.target.value })}
                                        />
                                    </div>
                                </div>

                                <div className="flex justify-end space-x-4 pt-4">
                                    <button
                                        type="button"
                                        onClick={() => setIsChangePasswordOpen(false)}
                                        className="btn-secondary py-2.5 px-6"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        type="submit"
                                        className="btn-primary"
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
