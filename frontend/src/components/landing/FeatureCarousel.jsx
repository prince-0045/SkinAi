import React from 'react';
import { motion } from 'framer-motion';
import { Activity, MapPin, Shield, TrendingUp } from 'lucide-react';
import { Link } from 'react-router-dom';

const features = [
    {
        id: 1,
        title: "AI Detection",
        icon: Activity,
        description: "Instant skin analysis powered by advanced machine learning.",
        link: "/detect",
        color: "bg-blue-500"
    },
    {
        id: 2,
        title: "Find Doctors",
        icon: MapPin,
        description: "Locate verified dermatologists near you in seconds.",
        link: "/doctors",
        color: "bg-green-500"
    },
    {
        id: 3,
        title: "History",
        icon: TrendingUp,
        description: "View your past scans and skin recovery progress.",
        link: "/track",
        color: "bg-purple-500"
    },
    {
        id: 4,
        title: "Secure Data",
        icon: Shield,
        description: "Your health records are encrypted and private.",
        link: "/profile",
        color: "bg-indigo-500"
    }
];

// Duplicate list for seamless infinite scroll
const carouselItems = [...features, ...features, ...features];

export default function FeatureCarousel() {
    return (
        <div className="w-full overflow-hidden py-10 bg-gray-50 dark:bg-slate-900/50">
            <div className="max-w-7xl mx-auto px-4 mb-6">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white text-center">Everything You Need</h2>
            </div>

            <div className="relative w-full flex">
                <motion.div
                    className="flex space-x-6 whitespace-nowrap"
                    animate={{ x: ["0%", "-33.33%"] }}
                    transition={{
                        repeat: Infinity,
                        ease: "linear",
                        duration: 20 // Adjust speed here
                    }}
                >
                    {carouselItems.map((item, index) => (
                        <Link
                            key={`${item.id}-${index}`}
                            to={item.link}
                            className="inline-block w-72 h-48 bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-gray-100 dark:border-slate-700 p-6 hover:shadow-md transition-shadow flex-shrink-0 whitespace-normal"
                        >
                            <div className={`w-12 h-12 ${item.color} rounded-xl flex items-center justify-center mb-4 text-white shadow-lg`}>
                                <item.icon className="w-6 h-6" />
                            </div>
                            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">{item.title}</h3>
                            <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">
                                {item.description}
                            </p>
                        </Link>
                    ))}
                </motion.div>

                {/* Gradient Masks for fade effect */}
                <div className="absolute top-0 left-0 h-full w-12 bg-gradient-to-r from-gray-50 dark:from-slate-900 to-transparent pointer-events-none"></div>
                <div className="absolute top-0 right-0 h-full w-12 bg-gradient-to-l from-gray-50 dark:from-slate-900 to-transparent pointer-events-none"></div>
            </div>
        </div>
    );
}
