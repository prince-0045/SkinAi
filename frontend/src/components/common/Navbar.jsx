import React from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { Activity, Menu, X, User } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import DayNightToggle from '../DayNightToggle';
import DermAuraLogo from './DermAuraLogo';

export default function Navbar() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();
    const [isOpen, setIsOpen] = React.useState(false);

    // Dynamic styles for active navigation items
    const navClass = (path) => `nav-link ${location.pathname === path ? 'active' : ''}`;
    const mobClass = (path) => `block px-3 py-2 rounded-md text-base font-medium transition-colors ${location.pathname === path ? 'text-[var(--blue-bright)] bg-[var(--blue-dim)]' : 'text-[var(--white-muted)] hover:text-white hover:bg-[var(--bg-card)]'}`;

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    return (
        <nav className="navbar">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16 items-center">
                    {/* Logo */}
                    <Link to="/" className="flex items-center">
                        <DermAuraLogo className="h-12 md:h-14 w-auto hover:opacity-90 transition-opacity" />
                    </Link>

                    {/* Desktop Nav */}
                    <div className="hidden md:flex items-center space-x-8">
                        {user && (
                            <Link to="/" className={navClass('/')}>
                                Home
                            </Link>
                        )}
                        <Link to="/track" className={navClass('/track')}>
                            History
                        </Link>
                        <Link to="/detect" className={navClass('/detect')}>
                            AI Detection
                        </Link>
                        <Link to="/doctors" className={navClass('/doctors')}>
                            Find Doctors
                        </Link>

                        <DayNightToggle />

                        {user ? (
                            <div className="flex items-center space-x-4">
                                <Link to="/profile" className="p-2 rounded-full hover:bg-[var(--blue-dim)] transition-colors">
                                    <User className="h-5 w-5 text-white" />
                                </Link>
                                <button
                                    onClick={handleLogout}
                                    className="btn-secondary px-4 py-2 text-sm"
                                >
                                    Logout
                                </button>
                            </div>
                        ) : (
                            <div className="flex items-center space-x-4">
                                <Link to="/login" className="nav-link">
                                    Login
                                </Link>
                                <Link to="/signup" className="btn-primary px-5 py-2.5">
                                    Get Started
                                </Link>
                            </div>
                        )}
                    </div>

                    {/* Mobile menu button */}
                    <div className="md:hidden">
                        <button
                            onClick={() => setIsOpen(!isOpen)}
                            className="p-2 rounded-md text-white hover:bg-[var(--blue-dim)]"
                        >
                            {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
                        </button>
                    </div>
                </div>
            </div>

            {/* Mobile Menu */}
            {isOpen && (
                <div className="md:hidden bg-[var(--bg-surface)] border-b border-[var(--border-subtle)]">
                    <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
                        <Link to="/" className={mobClass('/')}>Home</Link>
                        <Link to="/track" className={mobClass('/track')}>History</Link>
                        <Link to="/detect" className={mobClass('/detect')}>Detection</Link>
                        <Link to="/doctors" className={mobClass('/doctors')}>Find Doctors</Link>
                        <div className="px-3 py-2">
                            <DayNightToggle />
                        </div>
                        {user ? (
                            <>
                                <Link to="/profile" className="block px-3 py-2 rounded-md text-base font-medium text-[var(--white-muted)] hover:text-white hover:bg-[var(--bg-card)]">Profile</Link>
                                <button onClick={handleLogout} className="w-full text-left block px-3 py-2 rounded-md text-base font-medium text-[var(--blue-soft)] hover:bg-[var(--blue-dim)]">Logout</button>
                            </>
                        ) : (
                            <>
                                <Link to="/login" className="block px-3 py-2 rounded-md text-base font-medium text-[var(--white-muted)] hover:text-white hover:bg-[var(--bg-card)]">Login</Link>
                                <Link to="/signup" className="block px-3 py-2 rounded-md text-base font-medium text-[var(--blue-bright)] hover:bg-[var(--blue-dim)]">Sign Up</Link>
                            </>
                        )}
                    </div>
                </div>
            )}
        </nav>
    );
}
