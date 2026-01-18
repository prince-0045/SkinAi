import React from 'react';
import { Helmet } from 'react-helmet-async';
import { motion } from 'framer-motion';
import HealingChart from '../components/dashboard/HealingChart';
import ImageComparison from '../components/dashboard/ImageComparison';
import { Calendar, TrendingUp, AlertCircle, Clock } from 'lucide-react';

export default function Tracker() {
    const dummyBefore = "https://images.unsplash.com/photo-1614859324967-dadb3e51d939?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80"; // Red skin example
    const dummyAfter = "https://images.unsplash.com/photo-1512413914633-b5043f4041ea?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80"; // Clear skin example

    const timelineEvents = [
        { date: 'Oct 24', title: 'Initial Scan', status: 'Detected', severity: 'High' },
        { date: 'Oct 27', title: 'Follow-up', status: 'Improving', severity: 'Medium' },
        { date: 'Nov 04', title: 'Routine Check', status: 'Clear', severity: 'Low' },
    ];

    return (
        <div className="min-h-screen bg-gray-50 pt-24 pb-12">
            <Helmet>
                <title>Healing Tracker - SkinAi</title>
            </Helmet>

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-900">Healing Dashboard</h1>
                    <p className="text-gray-600">Track your recovery progress and monitor skin health statistics.</p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Main Stats Column */}
                    <div className="lg:col-span-2 space-y-8">
                        {/* Healing Chart */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100"
                        >
                            <div className="flex items-center justify-between mb-6">
                                <h2 className="text-lg font-bold text-gray-900 flex items-center">
                                    <TrendingUp className="w-5 h-5 mr-2 text-medical-600" />
                                    Recovery Trajectory
                                </h2>
                                <span className="text-sm text-green-600 font-semibold bg-green-50 px-3 py-1 rounded-full">
                                    +42% Improvement
                                </span>
                            </div>
                            <HealingChart />
                        </motion.div>

                        {/* Before vs After */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1 }}
                            className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100"
                        >
                            <h2 className="text-lg font-bold text-gray-900 mb-4">Visual Progress</h2>
                            <ImageComparison beforeImage={dummyBefore} afterImage={dummyAfter} />
                            <p className="mt-4 text-sm text-gray-500 text-center">Drag the slider to compare initial scan vs current status.</p>
                        </motion.div>
                    </div>

                    {/* Sidebar / Timeline */}
                    <div className="space-y-8">
                        {/* Current Status Card */}
                        <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="bg-gradient-to-br from-medical-600 to-medical-800 rounded-2xl p-6 text-white shadow-lg"
                        >
                            <h3 className="text-lg font-semibold mb-2">Current Status</h3>
                            <div className="text-4xl font-bold mb-4">Healing</div>
                            <div className="space-y-2">
                                <div className="flex justify-between text-sm text-medical-100">
                                    <span>Inflammation</span>
                                    <span>Low</span>
                                </div>
                                <div className="w-full bg-white/20 rounded-full h-2">
                                    <div className="bg-green-400 h-2 rounded-full" style={{ width: '20%' }}></div>
                                </div>
                            </div>
                        </motion.div>

                        {/* Timeline */}
                        <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                            <h3 className="text-lg font-bold text-gray-900 mb-6 flex items-center">
                                <Calendar className="w-5 h-5 mr-2 text-gray-400" />
                                History
                            </h3>
                            <div className="space-y-8 pl-2">
                                {timelineEvents.map((event, index) => (
                                    <div key={index} className="relative pl-8 border-l-2 border-gray-200 last:border-0 pb-2">
                                        <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-medical-500 ring-4 ring-white"></div>
                                        <div>
                                            <div className="text-xs text-gray-500 font-medium uppercase tracking-wider mb-1">{event.date}</div>
                                            <div className="font-semibold text-gray-900">{event.title}</div>
                                            <div className="text-sm text-gray-600 mt-1 flex items-center">
                                                Status:
                                                <span className={`ml-2 px-2 py-0.5 rounded text-xs font-semibold 
                                            ${event.status === 'Detected' ? 'bg-red-100 text-red-700' :
                                                        event.status === 'Improving' ? 'bg-yellow-100 text-yellow-700' : 'bg-green-100 text-green-700'}`}>
                                                    {event.status}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
