import React from 'react';
import { Helmet } from 'react-helmet-async';
import Hero from '../components/landing/Hero';
// import Features from '../components/landing/Features'; // To be created
// import Testimonials from '../components/landing/Testimonials'; // To be created
import { motion } from 'framer-motion';
import { Activity, Clock, Shield, Upload } from 'lucide-react';

const FeatureCard = ({ icon: Icon, title, description, delay }) => (
    <motion.div
        className="relative p-6 bg-white rounded-2xl border border-gray-100 shadow-sm hover:shadow-md transition-shadow"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ delay, duration: 0.5 }}
    >
        <div className="w-12 h-12 bg-medical-50 rounded-xl flex items-center justify-center mb-4 text-medical-600">
            <Icon className="w-6 h-6" />
        </div>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">{title}</h3>
        <p className="text-gray-500">{description}</p>
    </motion.div>
);

export default function Landing() {
    return (
        <div className="min-h-screen bg-white">
            <Helmet>
                <title>SkinAi - Advanced Skin Disease Detection</title>
                <meta name="description" content="AI-powered skin disease detection and healing tracker." />
            </Helmet>

            <Hero />

            {/* Features Section - Inline for now or separate later */}
            <section className="py-24 bg-white">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-bold text-gray-900 sm:text-4xl">Why Choose SkinAi?</h2>
                        <p className="mt-4 text-lg text-gray-500">Comprehensive dermatological analysis at your fingertips</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
                        <FeatureCard
                            icon={Upload}
                            title="Instant Analysis"
                            description="Upload a photo and get AI-powered insights in seconds."
                            delay={0.1}
                        />
                        <FeatureCard
                            icon={Activity}
                            title="Healing Tracker"
                            description="Monitor your skin's recovery progress over time with visual graphs."
                            delay={0.2}
                        />
                        <FeatureCard
                            icon={Shield}
                            title="Privacy First"
                            description="Your medical data is encrypted and secure. We prioritize your privacy."
                            delay={0.3}
                        />
                        <FeatureCard
                            icon={Clock}
                            title="24/7 Availability"
                            description="Access medical-grade screening anytime, anywhere."
                            delay={0.4}
                        />
                    </div>
                </div>
            </section>

            {/* Testimonials or Stats */}
            <section className="py-24 bg-medical-900 text-white overflow-hidden relative">
                {/* Decorative Circles */}
                <div className="absolute top-0 left-0 w-64 h-64 bg-medical-800 rounded-full blur-3xl -translate-x-1/2 -translate-y-1/2 opacity-50"></div>

                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
                        <div className="p-4">
                            <div className="text-4xl font-bold text-medical-300 mb-2">98%</div>
                            <div className="text-medical-100">Detection Accuracy</div>
                        </div>
                        <div className="p-4">
                            <div className="text-4xl font-bold text-medical-300 mb-2">10k+</div>
                            <div className="text-medical-100">Scans Performed</div>
                        </div>
                        <div className="p-4">
                            <div className="text-4xl font-bold text-medical-300 mb-2">24/7</div>
                            <div className="text-medical-100">Peace of Mind</div>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    );
}
