import React, { useState, useEffect } from 'react';

export default function DayNightToggle() {
    const [isDark, setIsDark] = useState(false);

    useEffect(() => {
        // Check local storage or system preference
        const storedTheme = localStorage.getItem('theme');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

        if (storedTheme === 'dark' || (!storedTheme && prefersDark)) {
            setIsDark(true);
            document.documentElement.classList.add('dark');
        } else {
            setIsDark(false);
            document.documentElement.classList.remove('dark');
        }
    }, []);

    const toggleTheme = () => {
        const newIsDark = !isDark;
        setIsDark(newIsDark);

        if (newIsDark) {
            document.documentElement.classList.add('dark');
            localStorage.setItem('theme', 'dark');
        } else {
            document.documentElement.classList.remove('dark');
            localStorage.setItem('theme', 'light');
        }
    };

    return (
        <button
            onClick={toggleTheme}
            className={`relative w-16 h-8 rounded-full p-1 transition-colors duration-500 ease-in-out focus:outline-none ${isDark ? 'bg-indigo-900' : 'bg-sky-400'}`}
            aria-label="Toggle Dark Mode"
        >
            <div
                className={`absolute top-1 left-1 w-6 h-6 rounded-full transform transition-transform duration-500 ease-in-out ${isDark ? 'translate-x-8 bg-gray-100' : 'translate-x-0 bg-yellow-300'}`}
            >
                {/* Sun Icon Details */}
                {!isDark && (
                    <span className="absolute inset-0 flex items-center justify-center animate-pulse-slow">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" className="w-4 h-4 text-orange-400">
                            <circle cx="12" cy="12" r="4" fill="currentColor" className="opacity-80" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32l1.41 1.41M2 12h2m16 0h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
                        </svg>
                    </span>
                )}

                {/* Moon Icon Details */}
                {isDark && (
                    <span className="absolute inset-0 flex items-center justify-center">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" className="w-4 h-4 text-gray-800">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={0} fill="currentColor" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                        </svg>
                        {/* Craters */}
                        <span className="absolute top-1 right-2 w-1 h-1 bg-gray-300 rounded-full opacity-50"></span>
                        <span className="absolute bottom-2 left-2 w-1.5 h-1.5 bg-gray-300 rounded-full opacity-50"></span>
                    </span>
                )}
            </div>

            {/* Background Elements */}
            {!isDark && (
                <>
                    <span className="absolute top-2 right-4 text-white opacity-80 pointer-events-none">☁️</span>
                    <span className="absolute bottom-1 left-8 text-white opacity-60 text-xs pointer-events-none">☁️</span>
                </>
            )}

            {isDark && (
                <>
                    <span className="absolute top-2 left-3 text-white text-[8px] animate-pulse">★</span>
                    <span className="absolute bottom-2 right-5 text-white text-[6px] animate-pulse delay-75">★</span>
                    <span className="absolute top-1 right-3 text-white text-[7px] animate-pulse delay-150">★</span>
                </>
            )}
        </button>
    );
}
