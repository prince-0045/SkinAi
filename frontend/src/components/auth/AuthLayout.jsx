import React from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, ShieldCheck } from 'lucide-react';
import { Link } from 'react-router-dom';
import DNAHelix from '../animations/DNAHelix';

export default function AuthLayout({ children, title, subtitle }) {
    return (
        <div className="bg-[#0d1117] bg-navy-mesh min-h-screen flex transition-colors duration-300">
            {/* Left Side - Form */}
            <div className="bg-[#0d1117] bg-navy-mesh w-full lg:w-1/2 flex flex-col justify-center px-4 sm:px-12 md:px-24 relative">

                <motion.div
                    className="relative z-10 pt-32 lg:pt-24"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    <div className="mb-8">
                        <div className="inline-flex items-center px-3 py-1 bg-medical-50 dark:bg-medical-900/30 rounded-full text-medical-600 dark:text-medical-400 text-sm font-semibold mb-4 border border-medical-100 dark:border-medical-800">
                            <ShieldCheck className="w-4 h-4 mr-2" />
                            Secure Medical Login
                        </div>
                        <h2 className="text-3xl font-bold text-gray-900 dark:text-white">{title}</h2>
                        <p className="mt-2 text-gray-600 dark:text-gray-400">{subtitle}</p>
                    </div>

                    {children}
                </motion.div>
            </div>

            {/* Right Side - Visual */}
            <div className="bg-[#0d1117] bg-navy-mesh hidden lg:flex w-1/2 relative overflow-hidden items-center justify-center p-12">
                <div className="absolute inset-0 bg-gradient-to-br from-medical-900/40 to-slate-950 opacity-100"></div>
                {/* Decorative DNA */}
                <div className="absolute inset-0 opacity-20 transform scale-150 rotate-45 pointer-events-none">
                    <DNAHelix className="w-full h-full text-white" />
                </div>
                
                <div className="relative z-10 w-full flex flex-col items-center">
                    <motion.div 
                        className="text-center bg-slate-900/50 backdrop-blur-md p-8 rounded-2xl border border-white/10"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                    >
                        <h3 className="text-3xl font-bold text-white mb-4">Advanced AI Analysis</h3>
                        <p className="text-gray-400 text-lg max-w-sm mx-auto">Join thousands of users tracking their healing journey with clinical-grade precision.</p>
                        
                        <div className="mt-8 grid grid-cols-2 gap-4 text-left">
                            <div className="bg-white/5 p-4 rounded-xl border border-white/10">
                                <div className="text-2xl font-bold text-white mb-1">98%</div>
                                <div className="text-sm text-gray-400">Accuracy Rate</div>
                            </div>
                            <div className="bg-white/5 p-4 rounded-xl border border-white/10">
                                <div className="text-2xl font-bold text-white mb-1">24/7</div>
                                <div className="text-sm text-gray-400">Accessible Analysis</div>
                            </div>
                        </div>
                    </motion.div>
                </div>
            </div>
        </div>
    );
}
