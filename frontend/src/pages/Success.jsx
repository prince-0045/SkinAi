import React, { useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { CheckCircle, ArrowRight } from 'lucide-react';
import confetti from 'canvas-confetti';

export default function Success() {
    const navigate = useNavigate();

    useEffect(() => {
        // Fire confetti on mount
        const duration = 3 * 1000;
        const end = Date.now() + duration;

        const frame = () => {
            confetti({
                particleCount: 2,
                angle: 60,
                spread: 55,
                origin: { x: 0 },
                colors: ['#378ADD', '#60a5e8']
            });
            confetti({
                particleCount: 2,
                angle: 120,
                spread: 55,
                origin: { x: 1 },
                colors: ['#378ADD', '#60a5e8']
            });

            if (Date.now() < end) {
                requestAnimationFrame(frame);
            }
        };

        frame();
    }, []);

    return (
        <div className="bg-base min-h-screen transition-colors duration-300 flex items-center justify-center px-4">
            <motion.div
                className="max-w-md w-full card p-8 text-center"
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: "spring", duration: 0.6 }}
            >
                <div className="mx-auto w-20 h-20 bg-[var(--blue-dim)] rounded-full flex items-center justify-center mb-6">
                    <CheckCircle className="w-10 h-10 text-[var(--blue-bright)]" />
                </div>

                <h2 className="text-3xl font-bold text-white mb-2" style={{ fontFamily: 'var(--font-display)' }}>Welcome Aboard!</h2>
                <p className="text-[var(--white-muted)] mb-8">
                    Your account has been successfully verified. You now have access to our AI detection and healing tracker.
                </p>

                <div className="space-y-3">
                    <Link
                        to="/"
                        className="btn-primary w-full flex justify-center py-3"
                    >
                        Start Diagnosis <ArrowRight className="ml-2 w-4 h-4" />
                    </Link>
                    <Link
                        to="/"
                        className="btn-secondary w-full flex justify-center py-3"
                    >
                        Go to Home
                    </Link>
                </div>
            </motion.div>
        </div>
    );
}
