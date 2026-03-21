import React from 'react';
import { motion } from 'framer-motion';

export default function ScanningLoader() {
    return (
        <div className="absolute inset-0 flex flex-col items-center justify-end overflow-hidden">
            {/* Scanning Line — sweeps top to bottom continuously */}
            <motion.div
                className="absolute left-0 w-full h-0.5 z-10"
                style={{
                    background: 'linear-gradient(90deg, transparent 0%, #38bdf8 20%, #0ea5e9 50%, #38bdf8 80%, transparent 100%)',
                    boxShadow: '0 0 18px 4px rgba(14,165,233,0.6), 0 0 60px 8px rgba(14,165,233,0.25)',
                }}
                animate={{ top: ['-2%', '102%'] }}
                transition={{ duration: 2.2, repeat: Infinity, ease: 'linear' }}
            />

            {/* Subtle full-area tint so text is readable */}
            <div className="absolute inset-0 bg-black/30 pointer-events-none" />

            {/* Analyzing text at the bottom */}
            <div className="relative z-20 mb-6 text-center">
                <motion.span
                    className="text-cyan-300 font-mono text-sm tracking-widest drop-shadow-[0_0_6px_rgba(56,189,248,0.7)]"
                    animate={{ opacity: [0.4, 1, 0.4] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                >
                    ANALYZING SKIN TEXTURE...
                </motion.span>
            </div>
        </div>
    );
}
