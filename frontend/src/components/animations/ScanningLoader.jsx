import React from 'react';
import { motion } from 'framer-motion';

export default function ScanningLoader() {
    return (
        <div className="relative w-full h-64 bg-gray-900 rounded-lg overflow-hidden flex items-center justify-center">
            {/* Grid Background */}
            <div className="absolute inset-0 opacity-20"
                style={{ backgroundImage: 'linear-gradient(#0ea5e9 1px, transparent 1px), linear-gradient(90deg, #0ea5e9 1px, transparent 1px)', backgroundSize: '40px 40px' }}
            />

            {/* Scanning Line */}
            <motion.div
                className="absolute w-full h-1 bg-medical-400 shadow-[0_0_20px_rgba(56,189,248,0.8)] z-10"
                animate={{ top: ['0%', '100%'], opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            />

            {/* Central Medical Icon */}
            <div className="relative z-0 text-medical-500 animate-pulse">
                <svg className="w-16 h-16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
            </div>

            <div className="absolute bottom-4 left-0 right-0 text-center text-medical-300 font-mono text-sm">
                <motion.span animate={{ opacity: [0, 1, 0] }} transition={{ duration: 1.5, repeat: Infinity }}>
                    ANALYZING SKIN TEXTURE...
                </motion.span>
            </div>
        </div>
    );
}
