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
                colors: ['#0ea5e9', '#10b981']
            });
            confetti({
                particleCount: 2,
                angle: 120,
                spread: 55,
                origin: { x: 1 },
                colors: ['#0ea5e9', '#10b981']
            });

            if (Date.now() < end) {
                requestAnimationFrame(frame);
            }
        };

        frame();
    }, []);

    return (
        <div className="min-h-screen bg-medical-50 dark:bg-slate-900 transition-colors duration-300 flex items-center justify-center px-4">
            <motion.div
                className="max-w-md w-full bg-white rounded-2xl shadow-xl p-8 text-center"
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: "spring", duration: 0.6 }}
            >
                <div className="mx-auto w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mb-6">
                    <CheckCircle className="w-10 h-10 text-green-600" />
                </div>

                <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Welcome Aboard!</h2>
                <p className="text-gray-600 dark:text-gray-400 mb-8">
                    Your account has been successfully verified. You now have access to our AI detection and healing tracker.
                </p>

                <div className="space-y-3">
                    <Link
                        to="/detect"
                        className="w-full flex justify-center items-center py-3 px-4 rounded-lg bg-medical-600 text-white font-medium hover:bg-medical-700 transition"
                    >
                        Start Diagnosis <ArrowRight className="ml-2 w-4 h-4" />
                    </Link>
                    <Link
                        to="/"
                        className="w-full flex justify-center items-center py-3 px-4 rounded-lg border border-gray-200 dark:border-slate-600 text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-slate-700 transition"
                    >
                        Go to Home
                    </Link>
                </div>
            </motion.div>
        </div>
    );
}
