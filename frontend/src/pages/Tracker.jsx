import React, { useState, useEffect } from 'react';
import { Helmet } from 'react-helmet-async';
import { motion } from 'framer-motion';
import { Calendar, Activity, AlertCircle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function Tracker() {
    const { user } = useAuth();
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const token = localStorage.getItem('token');
                const response = await fetch('/api/v1/scan/history', {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                });

                if (response.ok) {
                    const data = await response.json();
                    setHistory(data);
                } else {
                    console.error("Failed to fetch history");
                    setError("Failed to load history");
                }
            } catch (err) {
                console.error("Error fetching history:", err);
                setError("Failed to load history");
            } finally {
                setLoading(false);
            }
        };

        fetchHistory();
    }, []);

    const formatDate = (dateString) => {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-slate-900 transition-colors duration-300 pt-24 pb-12">
            <Helmet>
                <title>History - SkinAi</title>
            </Helmet>

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Scan History</h1>
                    <p className="text-gray-600 dark:text-gray-400">Archive of your past skin analyses.</p>
                </div>

                {loading ? (
                    <div className="flex justify-center items-center h-64">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-medical-600"></div>
                    </div>
                ) : error ? (
                    <div className="bg-red-50 p-4 rounded-lg text-red-700 text-center">
                        {error}
                    </div>
                ) : history.length === 0 ? (
                    <div className="text-center py-12 bg-white rounded-2xl shadow-sm border border-gray-100">
                        <Activity className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                        <h3 className="text-lg font-medium text-gray-900">No scans yet</h3>
                        <p className="text-gray-500 mt-2">Upload your first image to get started.</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                        {history.map((scan, index) => (
                            <motion.div
                                key={scan.id || index}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: index * 0.1 }}
                                className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-gray-100 dark:border-slate-700 overflow-hidden hover:shadow-md transition-shadow"
                            >
                                <div className="aspect-video relative overflow-hidden bg-gray-100 group">
                                    <img
                                        src={scan.image_url}
                                        alt={`Scan ${index + 1}`}
                                        className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                                    />
                                    <div className="absolute top-2 right-2">
                                        <span className={`px-2 py-1 rounded-full text-xs font-semibold shadow-sm
                                            ${scan.severity_level === 'Severe' ? 'bg-red-100 text-red-700' :
                                                scan.severity_level === 'Moderate' ? 'bg-yellow-100 text-yellow-700' :
                                                    'bg-green-100 text-green-700'}`}>
                                            {scan.severity_level}
                                        </span>
                                    </div>
                                </div>
                                <div className="p-5">
                                    <div className="flex justify-between items-start mb-3">
                                        <div>
                                            <h3 className="font-bold text-gray-900 dark:text-white text-lg">{scan.disease_detected}</h3>
                                            <div className="flex items-center text-sm text-gray-500 dark:text-gray-400 mt-1">
                                                <Calendar className="w-4 h-4 mr-1.5" />
                                                {formatDate(scan.created_at)}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="border-t border-gray-100 pt-3 mt-3">
                                        <div className="flex justify-between items-center text-sm">
                                            <span className="text-gray-600">AI Confidence</span>
                                            <span className="font-semibold text-medical-600">
                                                {(scan.confidence_score * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                        <div className="w-full bg-gray-100 rounded-full h-1.5 mt-2">
                                            <div
                                                className="bg-medical-600 h-1.5 rounded-full"
                                                style={{ width: `${scan.confidence_score * 100}%` }}
                                            ></div>
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
