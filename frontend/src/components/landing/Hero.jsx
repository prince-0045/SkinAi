import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, ShieldCheck, Activity } from 'lucide-react';
import { motion } from 'framer-motion';
import DNAHelix from '../animations/DNAHelix';

export default function Hero() {
    return (
        <div className="relative overflow-hidden bg-gradient-to-b from-medical-50 to-white pt-16 pb-32">
            {/* Abstract Background Shapes */}
            <div className="absolute top-0 right-0 -translate-y-12 translate-x-12 opacity-10">
                <DNAHelix className="w-96 h-96" />
            </div>

            <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20">
                <div className="lg:grid lg:grid-cols-12 lg:gap-8">

                    <motion.div
                        className="sm:text-center md:max-w-2xl md:mx-auto lg:col-span-6 lg:text-left"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8 }}
                    >
                        <div className="inline-flex items-center px-4 py-2 rounded-full bg-medical-50 text-medical-600 text-sm font-semibold mb-6 border border-medical-100">
                            <ShieldCheck className="w-4 h-4 mr-2" />
                            AI-Powered Dermatology
                        </div>
                        <h1 className="text-4xl tracking-tight font-extrabold text-gray-900 sm:text-5xl md:text-6xl lg:text-5xl xl:text-6xl">
                            <span className="block">Advanced Skin Care</span>
                            <span className="block text-medical-600">Powered by AI</span>
                        </h1>
                        <p className="mt-3 text-base text-gray-500 sm:mt-5 sm:text-lg sm:max-w-xl sm:mx-auto md:mt-5 md:text-xl lg:mx-0">
                            Detect skin conditions instantly, track healing progress, and get personalized insights with our medical-grade AI technology.
                        </p>
                        <div className="mt-8 sm:max-w-lg sm:mx-auto sm:text-center lg:text-left lg:mx-0">
                            <div className="flex flex-col sm:flex-row gap-4">
                                <Link
                                    to="/detect"
                                    className="inline-flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-full text-white bg-medical-600 hover:bg-medical-700 md:py-4 md:text-lg shadow-lg shadow-medical-500/30 transition-all hover:-translate-y-1"
                                >
                                    Analyze My Skin
                                    <ArrowRight className="ml-2 w-5 h-5" />
                                </Link>
                                <Link
                                    to="/track"
                                    className="inline-flex items-center justify-center px-8 py-3 border border-gray-200 text-base font-medium rounded-full text-gray-700 bg-white hover:bg-gray-50 md:py-4 md:text-lg transition-all"
                                >
                                    Track Healing
                                </Link>
                            </div>
                            <p className="mt-4 text-sm text-gray-400 flex items-center gap-2 sm:justify-center lg:justify-start">
                                <Activity className="w-4 h-4" />
                                Clinical-grade accuracy • Privacy first
                            </p>
                        </div>
                    </motion.div>

                    {/* Right Side Visual */}
                    <motion.div
                        className="mt-12 relative sm:max-w-lg sm:mx-auto lg:mt-0 lg:max-w-none lg:mx-0 lg:col-span-6 lg:flex lg:items-center"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.8, delay: 0.2 }}
                    >
                        <div className="relative mx-auto w-full rounded-lg lg:max-w-md">
                            <div className="relative block w-full bg-white rounded-2xl shadow-xl overflow-hidden border border-medical-50">
                                <img
                                    src="https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?ixlib=rb-1.2.1&auto=format&fit=crop&w=1050&q=80"
                                    alt="Medical Analysis"
                                    className="w-full h-full object-cover opacity-90"
                                />
                                {/* Floating Cards Overlays */}
                                <div className="absolute top-4 left-4 bg-white/90 backdrop-blur p-3 rounded-lg shadow-lg border border-medical-100 flex items-center gap-3">
                                    <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                                        <ShieldCheck className="w-6 h-6 text-green-600" />
                                    </div>
                                    <div>
                                        <p className="text-xs text-gray-500">Confidence Score</p>
                                        <p className="text-sm font-bold text-gray-900">98.5% Accuracy</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </div>
            </div>
        </div>
    );
}
