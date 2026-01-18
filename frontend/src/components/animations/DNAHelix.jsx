import React from 'react';
import { motion } from 'framer-motion';

export default function DNAHelix({ className }) {
    return (
        <div className={className}>
            <svg viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-full opacity-20 text-medical-200">
                <motion.path
                    d="M40 100 Q 70 40, 100 100 T 160 100"
                    stroke="currentColor"
                    strokeWidth="8"
                    strokeLinecap="round"
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 1 }}
                    transition={{ duration: 2, ease: "easeInOut", repeat: Infinity, repeatType: "reverse" }}
                />
                <motion.path
                    d="M40 100 Q 70 160, 100 100 T 160 100"
                    stroke="currentColor"
                    strokeWidth="8"
                    strokeLinecap="round"
                    className="text-medical-400"
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 1 }}
                    transition={{ duration: 2, ease: "easeInOut", repeat: Infinity, repeatType: "reverse", delay: 0.5 }}
                />
                {/* DNA Base Pairs */}
                {[1, 2, 3, 4, 5].map((i) => (
                    <motion.line
                        key={i}
                        x1={50 + i * 20}
                        y1="80"
                        x2={50 + i * 20}
                        y2="120"
                        stroke="currentColor"
                        strokeWidth="4"
                        strokeDasharray="4 4"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: [0, 1, 0] }}
                        transition={{ duration: 3, repeat: Infinity, delay: i * 0.2 }}
                    />
                ))}
            </svg>
        </div>
    );
}
