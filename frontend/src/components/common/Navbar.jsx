import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Activity, Menu, X, User } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import DayNightToggle from '../DayNightToggle';

export default function Navbar() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const [isOpen, setIsOpen] = React.useState(false);

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    return (
        <nav className="fixed top-0 w-full bg-white/80 dark:bg-slate-900/80 backdrop-blur-md z-50 border-b border-gray-100 dark:border-slate-800 transition-colors duration-300">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16 items-center">
                    {/* Logo */}
                    <Link to="/" className="flex items-center space-x-2">
                        <div className="p-2 bg-medical-500 rounded-lg">
                            <Activity className="h-6 w-6 text-white" />
                        </div>
                        <span className="text-xl font-bold bg-gradient-to-r from-medical-600 to-medical-400 bg-clip-text text-transparent">
                            SkinAi
                        </span>
                    </Link>

                    {/* Desktop Nav */}
                    <div className="hidden md:flex items-center space-x-8">
                        {user && (
                            <Link to="/" className="text-gray-600 dark:text-gray-300 hover:text-medical-600 font-medium transition-colors">
                                Home
                            </Link>
                        )}
                        <Link to="/track" className="text-gray-600 dark:text-gray-300 hover:text-medical-600 font-medium transition-colors">
                            History
                        </Link>
                        <Link to="/detect" className="text-gray-600 dark:text-gray-300 hover:text-medical-600 font-medium transition-colors">
                            AI Detection
                        </Link>
                        <Link to="/doctors" className="text-gray-600 dark:text-gray-300 hover:text-medical-600 font-medium transition-colors">
                            Find Doctors
                        </Link>

                        <DayNightToggle />

                        {user ? (
                            <div className="flex items-center space-x-4">
                                <Link to="/profile" className="p-2 rounded-full hover:bg-gray-100 transition-colors">
                                    <User className="h-5 w-5 text-gray-700" />
                                </Link>
                                <button
                                    onClick={handleLogout}
                                    className="px-4 py-2 text-sm font-medium text-red-600 hover:bg-red-50 rounded-md transition-colors"
                                >
                                    Logout
                                </button>
                            </div>
                        ) : (
                            <div className="flex items-center space-x-4">
                                <Link to="/login" className="text-gray-600 hover:text-medical-600 font-medium transition-colors">
                                    Login
                                </Link>
                                <Link to="/signup" className="px-5 py-2.5 bg-medical-600 text-white rounded-full font-medium shadow-lg shadow-medical-500/30 hover:bg-medical-700 transition-all hover:-translate-y-0.5">
                                    Get Started
                                </Link>
                            </div>
                        )}
                    </div>

                    {/* Mobile menu button */}
                    <div className="md:hidden">
                        <button
                            onClick={() => setIsOpen(!isOpen)}
                            className="p-2 rounded-md text-gray-600 hover:bg-gray-100"
                        >
                            {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
                        </button>
                    </div>
                </div>
            </div>

            {/* Mobile Menu */}
            {isOpen && (
                <div className="md:hidden bg-white border-b border-gray-100">
                    <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
                        <Link to="/" className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 hover:text-medical-600 hover:bg-gray-50">Home</Link>
                        <Link to="/track" className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 hover:text-medical-600 hover:bg-gray-50">History</Link>
                        <Link to="/detect" className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 hover:text-medical-600 hover:bg-gray-50">Detection</Link>
                        <Link to="/doctors" className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 hover:text-medical-600 hover:bg-gray-50">Find Doctors</Link>
                        <div className="px-3 py-2">
                            <DayNightToggle />
                        </div>
                        {user ? (
                            <>
                                <Link to="/profile" className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 hover:text-medical-600 hover:bg-gray-50">Profile</Link>
                                <button onClick={handleLogout} className="w-full text-left block px-3 py-2 rounded-md text-base font-medium text-red-600 hover:bg-red-50">Logout</button>
                            </>
                        ) : (
                            <>
                                <Link to="/login" className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 hover:text-medical-600 hover:bg-gray-50">Login</Link>
                                <Link to="/signup" className="block px-3 py-2 rounded-md text-base font-medium text-medical-600 hover:bg-medical-50">Sign Up</Link>
                            </>
                        )}
                    </div>
                </div>
            )}
        </nav>
    );
}
