import React, { useState, useEffect, useMemo } from 'react';
import { Helmet } from 'react-helmet-async';
import { motion } from 'framer-motion';
import { Calendar, Activity, AlertCircle, BarChart3, Grid3X3 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { useAuth } from '../context/AuthContext';

// ── Skeleton component for loading state ──
const ScanCardSkeleton = () => (
    <div className="card overflow-hidden animate-pulse">
        <div className="aspect-video bg-[var(--blue-dim)] opacity-30" />
        <div className="p-5 space-y-3">
            <div className="h-5 bg-[var(--blue-dim)] opacity-40 rounded w-3/4" />
            <div className="h-4 bg-[var(--blue-dim)] opacity-40 rounded w-1/2" />
            <div className="border-t border-[var(--border-subtle)] pt-3 mt-3 space-y-2">
                <div className="flex justify-between">
                    <div className="h-4 bg-[var(--blue-dim)] opacity-40 rounded w-1/3" />
                    <div className="h-4 bg-[var(--blue-dim)] opacity-40 rounded w-1/6" />
                </div>
                <div className="h-1.5 bg-[var(--blue-dim)] opacity-40 rounded-full w-full" />
            </div>
        </div>
    </div>
);

// ── Custom Tooltip for Recharts ──
const ChartTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        return (
            <div className="card p-3 text-sm z-50 shadow-2xl">
                <p className="font-bold text-white" style={{ fontFamily: 'var(--font-display)' }}>{payload[0].payload.name}</p>
                <p className="text-[var(--white-muted)] text-xs mb-2">{payload[0].payload.dateDetailed}</p>
                <p className="text-[var(--blue-soft)]">
                    Confidence: <span className="font-bold text-[var(--blue-bright)]">{payload[0].value}%</span>
                </p>
                {payload[0].payload.condition && (
                    <p className="text-white mt-0.5 font-medium">
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
        <div className="bg-base min-h-screen transition-colors duration-300 pt-24 pb-12">
            <Helmet>
                <title>History - DERMAURA</title>
            </Helmet>

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 page-enter">
                {/* Header with view toggle */}
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-8 gap-4">
                    <div>
                        <h1 className="page-title text-3xl">Scan History</h1>
                        <p className="page-subtitle mt-2">Track your skin analyses over time.</p>
                    </div>
                    {!loading && history.length > 0 && (
                        <div className="flex card p-1" role="tablist" aria-label="View mode">
                            <button
                                onClick={() => setViewMode('grid')}
                                className={`view-toggle-btn ${viewMode === 'grid' ? 'active' : ''}`}
                                role="tab"
                                aria-selected={viewMode === 'grid'}
                                aria-controls="grid-view"
                            >
                                <Grid3X3 className="w-4 h-4" />
                                Grid
                            </button>
                            <button
                                onClick={() => setViewMode('chart')}
                                className={`view-toggle-btn ${viewMode === 'chart' ? 'active' : ''}`}
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
                    <div className="bg-[var(--blue-dim)] p-4 rounded-lg text-white border border-[var(--border-subtle)] text-center">
                        <AlertCircle className="w-5 h-5 inline mr-2 text-[var(--blue-bright)]" />
                        {error}
                    </div>
                ) : history.length === 0 ? (
                    <div className="text-center py-12 card !shadow-none">
                        <Activity className="w-12 h-12 text-[var(--blue-mid)] mx-auto mb-4 opacity-50" />
                        <h3 className="text-lg font-bold text-white" style={{ fontFamily: 'var(--font-display)' }}>No scans yet</h3>
                        <p className="text-[var(--white-muted)] mt-2">Upload your first image to get started.</p>
                    </div>
                ) : (
                    <>
                        {/* ── Stats summary cards ── */}
                        {stats && (
                            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="stat-card"
                                >
                                    <p className="stat-label">Total Scans</p>
                                    <p className="stat-value">{stats.totalScans}</p>
                                </motion.div>
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.1 }}
                                    className="stat-card"
                                >
                                    <p className="stat-label">Avg Confidence</p>
                                    <p className="stat-value blue">{stats.avgConfidence}%</p>
                                </motion.div>
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.2 }}
                                    className="stat-card"
                                >
                                    <p className="stat-label">Conditions Detected</p>
                                    <p className="stat-value">{stats.uniqueConditions}</p>
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
                                className="card p-6 mb-8"
                            >
                                <h2 className="text-lg font-bold text-white mb-4" style={{ fontFamily: 'var(--font-display)' }}>Confidence Over Time</h2>
                                <ResponsiveContainer width="100%" height={320}>
                                    <LineChart data={chartData} margin={{ top: 20, right: 20, left: 0, bottom: 40 }}>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(55,138,221,0.1)" />
                                        <XAxis
                                            dataKey="name"
                                            tick={{ fontSize: 11, fill: 'var(--white-muted)' }}
                                            axisLine={{ stroke: 'rgba(55,138,221,0.2)' }}
                                            angle={-35}
                                            textAnchor="end"
                                        />
                                        <YAxis
                                            domain={[0, 100]}
                                            tick={{ fontSize: 12, fill: 'var(--white-muted)' }}
                                            axisLine={{ stroke: 'rgba(55,138,221,0.2)' }}
                                            tickFormatter={(v) => `${v}%`}
                                        />
                                        <Tooltip content={<ChartTooltip />} cursor={{ stroke: 'rgba(55,138,221,0.2)', strokeWidth: 1 }} />
                                        <Line
                                            type="monotone"
                                            dataKey="confidence"
                                            stroke="#60a5e8"
                                            strokeWidth={3}
                                            dot={{ r: 4, fill: '#60a5e8', strokeWidth: 2 }}
                                            activeDot={{ r: 6 }}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            </motion.div>
                        )}

                        {viewMode === 'chart' && chartData.length <= 1 && (
                            <div className="card p-8 mb-8 text-center">
                                <BarChart3 className="w-10 h-10 text-[var(--blue-mid)] opacity-40 mx-auto mb-3" />
                                <p className="text-[var(--white-muted)]">Upload at least 2 scans to see trends over time.</p>
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
                                        className="scan-grid-card"
                                    >
                                        <div className="aspect-video thumbnail-wrap">
                                            <img
                                                src={scan.image_url}
                                                alt={`Scan for ${scan.disease_detected}`}
                                                className="w-full h-full object-cover"
                                                loading="lazy"
                                            />
                                            <div className="absolute top-2 right-2">
                                                <span className={
                                                    scan.severity_level === 'Critical' ? 'badge-positive !bg-[#1a3a6e] !text-white' :
                                                    scan.severity_level === 'High' ? 'badge-moderate !bg-[#378ADD] !text-white' :
                                                    'badge-moderate'
                                                }>
                                                    {scan.severity_level}
                                                </span>
                                            </div>
                                        </div>
                                        <div className="p-5">
                                            <div className="flex justify-between items-start mb-3">
                                                <div>
                                                    <h3 className="condition-name">{scan.disease_detected}</h3>
                                                    <div className="scan-date mt-1">
                                                        <Calendar className="w-4 h-4 mr-1" />
                                                        {formatDate(scan.created_at)}
                                                    </div>
                                                </div>
                                            </div>

                                            <div className="border-t border-[var(--border-subtle)] pt-3 mt-3">
                                                <div className="flex justify-between items-center text-sm mb-2">
                                                    <span className="text-[var(--white-faint)] font-medium" style={{ fontFamily: 'var(--font-display)' }}>AI Confidence</span>
                                                    <span className="font-bold text-[var(--blue-bright)]">
                                                        {(scan.confidence_score * 100).toFixed(1)}%
                                                    </span>
                                                </div>
                                                <div className="confidence-bar-track" role="progressbar" aria-valuenow={scan.confidence_score * 100} aria-valuemin="0" aria-valuemax="100">
                                                    <div
                                                        className="confidence-bar-fill"
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
