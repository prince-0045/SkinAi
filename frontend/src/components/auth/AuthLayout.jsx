import React from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, ShieldCheck } from 'lucide-react';
import { Link } from 'react-router-dom';
import DNAHelix from '../animations/DNAHelix';

export default function AuthLayout({ children, title, subtitle }) {
    return (
        <div className="min-h-screen flex bg-white">
            {/* Left Side - Form */}
            <div className="w-full lg:w-1/2 flex flex-col justify-center px-4 sm:px-12 md:px-24">
                <Link to="/" className="absolute top-8 left-8 text-gray-500 hover:text-medical-600 transition-colors flex items-center">
                    <ArrowLeft className="w-5 h-5 mr-1" /> Back to Home
                </Link>

                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    <div className="mb-8">
                        <div className="inline-flex items-center px-3 py-1 bg-medical-50 rounded-full text-medical-600 text-sm font-semibold mb-4">
                            <ShieldCheck className="w-4 h-4 mr-2" />
                            Secure Medical Login
                        </div>
                        <h2 className="text-3xl font-bold text-gray-900">{title}</h2>
                        <p className="mt-2 text-gray-600">{subtitle}</p>
                    </div>

                    {children}
                </motion.div>
            </div>

            {/* Right Side - Visual */}
            <div className="hidden lg:flex w-1/2 bg-medical-50 relative overflow-hidden items-center justify-center">
                <div className="absolute inset-0 bg-gradient-to-br from-medical-600 to-medical-900 opacity-90"></div>
                {/* Decorative DNA */}
                <div className="absolute inset-0 opacity-20 transform scale-150 rotate-45 pointer-events-none">
                    <DNAHelix className="w-full h-full text-white" />
                </div>

                <motion.div
                    className="relative z-10 text-white max-w-lg text-center p-12"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2, duration: 0.8 }}
                >
                    <h3 className="text-3xl font-bold mb-4">Advanced AI for Your Skin</h3>
                    <p className="text-medical-100 text-lg">Join thousands of users tracking their healing journey with clinical-grade precision.</p>

                    <div className="mt-12 grid grid-cols-2 gap-6 text-left">
                        <div className="bg-white/10 backdrop-blur p-4 rounded-xl border border-white/20">
                            <div className="text-2xl font-bold mb-1">98%</div>
                            <div className="text-sm text-medical-100">Accuracy Rate</div>
                        </div>
                        <div className="bg-white/10 backdrop-blur p-4 rounded-xl border border-white/20">
                            <div className="text-2xl font-bold mb-1">24/7</div>
                            <div className="text-sm text-medical-100">Accessible Analysis</div>
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
