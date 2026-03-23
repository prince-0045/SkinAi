import React, { useState, useEffect, useMemo } from 'react';
import { Helmet } from 'react-helmet-async';
import { motion } from 'framer-motion';
import { Calendar, Activity, AlertCircle, BarChart3, Grid3X3 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { useAuth } from '../context/AuthContext';

// ── Skeleton component for loading state ──
const ScanCardSkeleton = () => (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-gray-100 dark:border-slate-700 overflow-hidden animate-pulse">
        <div className="aspect-video bg-gray-200 dark:bg-slate-700" />
        <div className="p-5 space-y-3">
            <div className="h-5 bg-gray-200 dark:bg-slate-700 rounded w-3/4" />
            <div className="h-4 bg-gray-200 dark:bg-slate-700 rounded w-1/2" />
            <div className="border-t border-gray-100 dark:border-slate-700 pt-3 mt-3 space-y-2">
                <div className="flex justify-between">
                    <div className="h-4 bg-gray-200 dark:bg-slate-700 rounded w-1/3" />
                    <div className="h-4 bg-gray-200 dark:bg-slate-700 rounded w-1/6" />
                </div>
                <div className="h-1.5 bg-gray-200 dark:bg-slate-700 rounded-full w-full" />
            </div>
        </div>
    </div>
);

// ── Custom Tooltip for Recharts ──
const ChartTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-600 rounded-lg shadow-lg p-3 text-sm z-50">
                <p className="font-bold text-gray-900 dark:text-white">{payload[0].payload.name}</p>
                <p className="text-gray-400 text-xs mb-2">{payload[0].payload.dateDetailed}</p>
                <p className="text-medical-600">
                    Confidence: <span className="font-semibold">{payload[0].value}%</span>
                </p>
                {payload[0].payload.condition && (
                    <p className="text-gray-700 dark:text-gray-300 mt-0.5 font-medium">
                        {payload[0].payload.condition}
                    </p>
                )}
            </div>
        );
    }
    return null;
};

/**
 * Tracker Page Component
 * 
 * Displays the user's scan history in either a grid of cards or a timeline area chart.
 * Fetches historical scan data from the backend API using the user's JWT token.
 * Calculates summary statistics (total scans, average confidence, unique conditions) 
 * for quick insights.
 *
 * @component
 * @returns {React.ReactElement} The Tracker page view
 */
export default function Tracker() {
    const { user } = useAuth();
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [viewMode, setViewMode] = useState('grid'); // 'grid' or 'chart'

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const API_BASE_URL = import.meta.env.VITE_API_URL || '';
                const token = localStorage.getItem('token');
                const response = await fetch(`${API_BASE_URL}/api/v1/scan/history`, {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                });

                if (response.ok) {
                    const data = await response.json();
                    setHistory(data);
                } else {
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

    // ── Chart data (reverse chronological → chronological) ──
    const chartData = useMemo(() => {
        return [...history].reverse().map((scan, index) => ({
            name: `Scan ${index + 1}`,
            dateDetailed: new Date(scan.created_at).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
            confidence: parseFloat((scan.confidence_score * 100).toFixed(1)),
            condition: scan.disease_detected,
            severity: scan.severity_level,
        }));
    }, [history]);

    // ── Summary stats ──
    const stats = useMemo(() => {
        if (history.length === 0) return null;
        const avgConf = history.reduce((acc, s) => acc + s.confidence_score, 0) / history.length;
        const conditions = [...new Set(history.map(s => s.disease_detected))];
        return {
            totalScans: history.length,
            avgConfidence: (avgConf * 100).toFixed(1),
            uniqueConditions: conditions.length,
        };
    }, [history]);

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
        <div className="bg-[#0d1117] bg-navy-mesh min-h-screen transition-colors duration-300 pt-24 pb-12">
            <Helmet>
                <title>History - DERMAURA</title>
            </Helmet>

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                {/* Header with view toggle */}
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-8 gap-4">
                    <div>
                        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Scan History</h1>
                        <p className="text-gray-600 dark:text-gray-400">Track your skin analyses over time.</p>
                    </div>
                    {!loading && history.length > 0 && (
                        <div className="flex bg-white dark:bg-slate-800 rounded-lg border border-gray-200 dark:border-slate-700 p-1" role="tablist" aria-label="View mode">
                            <button
                                onClick={() => setViewMode('grid')}
                                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition ${
                                    viewMode === 'grid'
                                        ? 'bg-medical-600 text-white shadow-sm'
                                        : 'text-gray-500 hover:text-gray-900 dark:hover:text-white'
                                }`}
                                role="tab"
                                aria-selected={viewMode === 'grid'}
                                aria-controls="grid-view"
                            >
                                <Grid3X3 className="w-4 h-4" />
                                Grid
                            </button>
                            <button
                                onClick={() => setViewMode('chart')}
                                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition ${
                                    viewMode === 'chart'
                                        ? 'bg-medical-600 text-white shadow-sm'
                                        : 'text-gray-500 hover:text-gray-900 dark:hover:text-white'
                                }`}
                                role="tab"
                                aria-selected={viewMode === 'chart'}
                                aria-controls="chart-view"
                            >
                                <BarChart3 className="w-4 h-4" />
                                Timeline
                            </button>
                        </div>
                    )}
                </div>

                {loading ? (
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                        {[...Array(6)].map((_, i) => <ScanCardSkeleton key={i} />)}
                    </div>
                ) : error ? (
                    <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg text-red-700 dark:text-red-300 text-center border border-red-200 dark:border-red-800">
                        <AlertCircle className="w-5 h-5 inline mr-2" />
                        {error}
                    </div>
                ) : history.length === 0 ? (
                    <div className="text-center py-12 bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-gray-100 dark:border-slate-700">
                        <Activity className="w-12 h-12 text-gray-400 dark:text-gray-500 mx-auto mb-4" />
                        <h3 className="text-lg font-medium text-gray-900 dark:text-white">No scans yet</h3>
                        <p className="text-gray-500 dark:text-gray-400 mt-2">Upload your first image to get started.</p>
                    </div>
                ) : (
                    <>
                        {/* ── Stats summary cards ── */}
                        {stats && (
                            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="bg-white dark:bg-slate-800 rounded-xl border border-gray-100 dark:border-slate-700 p-5"
                                >
                                    <p className="text-sm text-gray-500 dark:text-gray-400">Total Scans</p>
                                    <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">{stats.totalScans}</p>
                                </motion.div>
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.1 }}
                                    className="bg-white dark:bg-slate-800 rounded-xl border border-gray-100 dark:border-slate-700 p-5"
                                >
                                    <p className="text-sm text-gray-500 dark:text-gray-400">Avg Confidence</p>
                                    <p className="text-2xl font-bold text-medical-600 mt-1">{stats.avgConfidence}%</p>
                                </motion.div>
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.2 }}
                                    className="bg-white dark:bg-slate-800 rounded-xl border border-gray-100 dark:border-slate-700 p-5"
                                >
                                    <p className="text-sm text-gray-500 dark:text-gray-400">Conditions Detected</p>
                                    <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">{stats.uniqueConditions}</p>
                                </motion.div>
                            </div>
                        )}

                        {/* ── Chart view ── */}
                        {viewMode === 'chart' && chartData.length > 1 && (
                            <motion.div
                                id="chart-view"
                                role="tabpanel"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="bg-white dark:bg-slate-800 rounded-xl border border-gray-100 dark:border-slate-700 p-6 mb-8"
                            >
                                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Confidence Over Time</h2>
                                <ResponsiveContainer width="100%" height={320}>
                                    <LineChart data={chartData} margin={{ top: 20, right: 20, left: 0, bottom: 40 }}>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                                        <XAxis
                                            dataKey="name"
                                            tick={{ fontSize: 11, fill: '#94a3b8' }}
                                            axisLine={{ stroke: '#e2e8f0' }}
                                            angle={-35}
                                            textAnchor="end"
                                        />
                                        <YAxis
                                            domain={[0, 100]}
                                            tick={{ fontSize: 12, fill: '#94a3b8' }}
                                            axisLine={{ stroke: '#e2e8f0' }}
                                            tickFormatter={(v) => `${v}%`}
                                        />
                                        <Tooltip content={<ChartTooltip />} cursor={{ stroke: 'rgba(0,0,0,0.1)', strokeWidth: 1 }} />
                                        <Line 
                                            type="monotone" 
                                            dataKey="confidence" 
                                            stroke="#0ea5e9"
                                            strokeWidth={3}
                                            dot={{ r: 4, fill: '#0ea5e9', strokeWidth: 2 }}
                                            activeDot={{ r: 6 }}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            </motion.div>
                        )}

                        {viewMode === 'chart' && chartData.length <= 1 && (
                            <div className="bg-white dark:bg-slate-800 rounded-xl border border-gray-100 dark:border-slate-700 p-8 mb-8 text-center">
                                <BarChart3 className="w-10 h-10 text-gray-300 dark:text-gray-600 mx-auto mb-3" />
                                <p className="text-gray-500 dark:text-gray-400">Upload at least 2 scans to see trends over time.</p>
                            </div>
                        )}

                        {/* ── Grid view ── */}
                        {viewMode === 'grid' && (
                            <div id="grid-view" role="tabpanel" className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                                {history.map((scan, index) => (
                                    <motion.div
                                        key={scan.id || index}
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: index * 0.05 }}
                                        className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-gray-100 dark:border-slate-700 overflow-hidden hover:shadow-md transition-shadow"
                                    >
                                        <div className="aspect-video relative overflow-hidden bg-gray-100 dark:bg-slate-700 group">
                                            <img
                                                src={scan.image_url}
                                                alt={`Scan for ${scan.disease_detected}`}
                                                className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                                                loading="lazy"
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

                                            <div className="border-t border-gray-100 dark:border-slate-700 pt-3 mt-3">
                                                <div className="flex justify-between items-center text-sm">
                                                    <span className="text-gray-600 dark:text-gray-400">AI Confidence</span>
                                                    <span className="font-semibold text-medical-600">
                                                        {(scan.confidence_score * 100).toFixed(1)}%
                                                    </span>
                                                </div>
                                                <div className="w-full bg-gray-100 dark:bg-slate-700 rounded-full h-1.5 mt-2" role="progressbar" aria-valuenow={scan.confidence_score * 100} aria-valuemin="0" aria-valuemax="100">
                                                    <div
                                                        className="bg-medical-600 h-1.5 rounded-full transition-all duration-500"
                                                        style={{ width: `${scan.confidence_score * 100}%` }}
                                                    ></div>
                                                </div>
                                            </div>
                                        </div>
                                    </motion.div>
                                ))}
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}
