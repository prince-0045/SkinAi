import React from 'react';
import { Helmet } from 'react-helmet-async';
import Hero from '../components/landing/Hero';
import FeatureCarousel from '../components/landing/FeatureCarousel';
// import Testimonials from '../components/landing/Testimonials'; // To be created
import { motion } from 'framer-motion';
import { Activity, Clock, Shield, Upload } from 'lucide-react';
import DNAHelix from '../components/animations/DNAHelix';

const FeatureCard = ({ icon: Icon, title, description, delay }) => (
    <motion.div
        className="card p-6"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ delay, duration: 0.5 }}
    >
        <div className="w-12 h-12 bg-[var(--blue-dim)] rounded-xl flex items-center justify-center mb-4 text-[var(--blue-bright)]">
            <Icon className="w-6 h-6" />
        </div>
        <h3 className="text-lg font-bold text-[var(--white-full)] mb-2" style={{ fontFamily: 'var(--font-display)' }}>{title}</h3>
        <p className="text-[var(--white-muted)]">{description}</p>
    </motion.div>
);

export default function Landing() {
    return (
        <div className="bg-base min-h-screen transition-colors duration-300">
            <Helmet>
                <title>DERMAURA - Advanced Skin Disease Detection</title>
                <meta name="description" content="AI-powered skin disease detection and healing tracker." />
            </Helmet>

            <Hero />
            <FeatureCarousel />

            {/* Features Section - Inline for now or separate later */}
            <section className="relative py-24 bg-[var(--bg-surface)] border-t border-[var(--border-subtle)] transition-colors duration-300 overflow-hidden">
                <div className="absolute -left-20 top-0 opacity-10 pointer-events-none">
                    <DNAHelix className="w-[600px] h-[600px] text-[#378ADD]" />
                </div>
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-bold text-[var(--white-full)] sm:text-4xl" style={{ fontFamily: 'var(--font-display)' }}>Why Choose DERMAURA?</h2>
                        <p className="mt-4 text-lg text-[var(--white-muted)]">Comprehensive dermatological analysis at your fingertips</p>
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


        </div>
    );
}
