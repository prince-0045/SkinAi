import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, AlertCircle, FileText, CheckCircle, ShieldAlert, ThumbsUp, ThumbsDown, Info } from 'lucide-react';
import ScanningLoader from '../components/animations/ScanningLoader';
import { useAuth } from '../context/AuthContext';
import { downloadMedicalReport } from '../utils/pdfGenerator';

const SEVERITY_CONFIG = {
    Low: { bg: 'bg-[#1a3a6e]/40', text: 'text-[var(--blue-soft)]', border: 'border-[var(--blue-mid)]', dot: 'bg-[var(--blue-bright)]' },
    Moderate: { bg: 'bg-[#1a3a6e]/60', text: 'text-[var(--white-soft)]', border: 'border-[var(--blue-mid)]', dot: 'bg-[var(--blue-bright)]' },
    High: { bg: 'bg-[#1a3a6e]/80', text: 'text-white', border: 'border-[#60a5e8]', dot: 'bg-white' },
    Critical: { bg: 'bg-[#378ADD]/30', text: 'text-white', border: 'border-white', dot: 'bg-white' },
    Unknown: { bg: 'bg-white/5', text: 'text-[var(--white-muted)]', border: 'border-[var(--white-faint)]', dot: 'bg-[var(--white-faint)]' },
};

export default function Detect() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [uploadLimit, setUploadLimit] = useState(null);

    useEffect(() => {
        const fetchLimit = async () => {
            try {
                const API_BASE_URL = import.meta.env.VITE_API_URL || '';
                const token = localStorage.getItem('token');
                const response = await fetch(`${API_BASE_URL}/api/v1/scan/limit`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                });
                if (response.ok) {
                    const data = await response.json();
                    setUploadLimit(data);
                }
            } catch (err) {
                console.error('Failed to fetch limit:', err);
            }
        };
        // Re-fetch limit on initial load and whenever an analysis officially completes (result changes)
        fetchLimit();
    }, [result]);

    const onDrop = useCallback(acceptedFiles => {
        const selectedFile = acceptedFiles[0];
        if (selectedFile) {
            setFile(selectedFile);
            setPreview(URL.createObjectURL(selectedFile));
            setResult(null);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'image/*': [] },
        maxFiles: 1
    });

    const clearFile = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
        setIsAnalyzing(false);
    };

    const { user } = useAuth();

    // #2: Compress image before upload — shrinks 5MB photos to ~100-200KB
    const compressImage = (imageFile, maxSize = 800, quality = 0.8) => {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new window.Image();
            img.onload = () => {
                let { width, height } = img;
                if (width > maxSize || height > maxSize) {
                    const ratio = Math.min(maxSize / width, maxSize / height);
                    width = Math.round(width * ratio);
                    height = Math.round(height * ratio);
                }
                canvas.width = width;
                canvas.height = height;
                ctx.drawImage(img, 0, 0, width, height);
                canvas.toBlob(
                    (blob) => resolve(new File([blob], imageFile.name, { type: 'image/jpeg' })),
                    'image/jpeg',
                    quality
                );
            };
            img.src = URL.createObjectURL(imageFile);
        });
    };

    const handleAnalyze = async () => {
        if (!file) return;
        setIsAnalyzing(true);
        setResult(null);
        setError(null);

        try {
            // Compress image before sending
            const compressedFile = await compressImage(file);

            const formData = new FormData();
            formData.append('file', compressedFile);

            const API_BASE_URL = import.meta.env.VITE_API_URL || '';
            const token = localStorage.getItem('token');
            const response = await fetch(`${API_BASE_URL}/api/v1/scan/predict`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` },
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                setResult({
                    disease: data.disease_detected,
                    confidence: (data.confidence_score * 100).toFixed(1),
                    severity: data.severity_level,
                    description: data.description || 'AI analysis complete based on the uploaded image.',
                    recommendation: data.recommendation || 'Please consult a dermatologist for a professional diagnosis.',
                    doList: data.do_list || [],
                    dontList: data.dont_list || [],
                });
            } else {
                const errorData = await response.json().catch(() => null);
                setError(errorData?.detail || 'Analysis failed. Please try again.');
            }
        } catch (err) {
            console.error('Error analyzing image:', err);
            setError('Error connecting to server.');
        } finally {
            setIsAnalyzing(false);
        }
    };

    const handleDownloadPDF = () => {
        downloadMedicalReport(result, user);
    };

    const sevCfg = result ? (SEVERITY_CONFIG[result.severity] || SEVERITY_CONFIG.Unknown) : null;

    return (
        <div className="bg-base min-h-screen transition-colors duration-300 pt-20 pb-12">
            <div className="max-w-5xl mx-auto px-4 sm:px-6 page-enter">
                <div className="text-center mb-10">
                    <h1 className="page-title text-3xl">AI Skin Disease Detection</h1>
                    <p className="page-subtitle mt-2">Upload a clear photo of the affected area for instant analysis.</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* ── Upload Area ── */}
                    <div className="card p-6">
                        {!preview ? (
                            <div
                                {...getRootProps()}
                                className={`upload-dropzone h-80 flex flex-col items-center justify-center cursor-pointer ${isDragActive ? 'dragging' : ''}`}
                            >
                                <input {...getInputProps()} />
                                <div className="upload-icon-circle mb-4">
                                    <Upload className="w-8 h-8" />
                                </div>
                                <p className="text-lg font-medium text-white" style={{ fontFamily: 'var(--font-display)' }}>Click to upload or drag &amp; drop</p>
                                <p className="text-sm text-[var(--white-muted)] mt-1">SVG, PNG, JPG or GIF (max. 5MB)</p>
                                {uploadLimit && (
                                    <div className="mt-4 badge-moderate inline-flex items-center gap-2">
                                        <Info className="w-3.5 h-3.5" />
                                        <span>{uploadLimit.remaining} of {uploadLimit.limit} daily uploads (resets 24h)</span>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="relative h-80 bg-black rounded-xl overflow-hidden flex items-center justify-center mb-4">
                                <img src={preview} alt="Preview" className="max-h-full max-w-full object-contain" />
                                {!isAnalyzing && (
                                    <button
                                        onClick={clearFile}
                                        className="absolute top-2 right-2 p-2 bg-black/50 text-white rounded-full hover:bg-black/70 transition"
                                    >
                                        <X className="w-5 h-5" />
                                    </button>
                                )}
                                {isAnalyzing && (
                                    <div className="absolute inset-0 bg-black/60 flex items-center justify-center z-10">
                                        <ScanningLoader />
                                    </div>
                                )}
                            </div>
                        )}

                        {error && (
                            <div className="mt-4 p-4 bg-[var(--blue-dim)] border border-[var(--border-subtle)] rounded-xl flex items-start gap-3">
                                <AlertCircle className="w-5 h-5 text-[var(--blue-bright)] flex-shrink-0 mt-0.5" />
                                <div>
                                    <h3 className="text-sm font-bold text-white" style={{ fontFamily: 'var(--font-display)' }}>Upload Notice</h3>
                                    <p className="text-sm text-[var(--white-muted)] mt-1">{error}</p>
                                </div>
                            </div>
                        )}

                        {preview && !isAnalyzing && !result && (
                            <button
                                onClick={handleAnalyze}
                                className="btn-primary w-full mt-5 py-3.5 text-base"
                            >
                                Analyze Image
                            </button>
                        )}
                    </div>

                    {/* ── Results Area ── */}
                    <div className="space-y-6">
                        <AnimatePresence>
                            {result ? (
                                <>
                                    {/* ── Detection Summary Card ── */}
                                    <motion.div
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="result-card p-6"
                                    >
                                        {/* Header */}
                                        <div className="flex items-center justify-between mb-4">
                                            <h2 className="text-xl font-bold text-white" style={{ fontFamily: 'var(--font-display)' }}>Analysis Result</h2>
                                            <span className="badge-positive w-max flex items-center gap-1">
                                                <CheckCircle className="w-3.5 h-3.5" /> Complete
                                            </span>
                                        </div>

                                        {/* Disease + Confidence */}
                                        <div className="mb-6">
                                            <p className="text-xs text-[var(--white-faint)] uppercase tracking-wide mb-1 font-bold">Detected Condition</p>
                                            <div className="detected-condition-name">{result.disease}</div>

                                            {/* Severity badge */}
                                            <span className={`inline-flex items-center gap-1.5 mt-2 px-3 py-1 rounded-full text-xs font-semibold border ${sevCfg.bg} ${sevCfg.text} ${sevCfg.border}`}>
                                                <span className={`w-2 h-2 rounded-full shadow-sm ${sevCfg.dot}`}></span>
                                                {result.severity} Severity
                                            </span>

                                            {/* Confidence bar */}
                                            <div className="mt-4">
                                                <div className="confidence-bar-track">
                                                    <div
                                                        className="confidence-bar-fill"
                                                        style={{ width: `${result.confidence}%` }}
                                                    />
                                                </div>
                                                <div className="flex justify-between mt-1.5 text-xs text-[var(--white-muted)]">
                                                    <span style={{ fontFamily: 'var(--font-display)', fontWeight: 600 }}>Confidence Score</span>
                                                    <span className="confidence-value">{result.confidence}%</span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Description */}
                                        <div className="info-box mb-4">
                                            <h3 className="info-box-title flex items-center gap-2 mb-2">
                                                <Info className="w-4 h-4" /> About This Condition
                                            </h3>
                                            <p className="text-sm text-[var(--white-muted)] leading-relaxed">{result.description}</p>
                                        </div>

                                        {/* Recommendation */}
                                        <div className="info-box">
                                            <h3 className="info-box-title flex items-center gap-2 mb-2">
                                                <AlertCircle className="w-4 h-4" /> Doctor's Advice
                                            </h3>
                                            <p className="text-sm text-[var(--white-full)] font-medium leading-relaxed">
                                                {result.recommendation}
                                            </p>
                                        </div>

                                        {/* Save button */}
                                        <div className="mt-5 pt-4 border-t border-[var(--border-subtle)] text-center">
                                            <button
                                                onClick={handleDownloadPDF}
                                                className="view-directions-link mx-auto text-sm"
                                            >
                                                <FileText className="w-4 h-4 mr-1" /> Save to Health Record
                                            </button>
                                        </div>
                                    </motion.div>

                                    {/* ── Do / Don't Cards ── */}
                                    {(result.doList.length > 0 || result.dontList.length > 0) && (
                                        <motion.div
                                            initial={{ opacity: 0, y: 20 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: 0.15 }}
                                            className="grid grid-cols-1 sm:grid-cols-2 gap-4"
                                        >
                                            {/* What To Do */}
                                            {result.doList.length > 0 && (
                                                <div className="action-card">
                                                    <div className="flex items-center gap-2 mb-3">
                                                        <ThumbsUp className="w-4 h-4 text-[var(--blue-bright)]" />
                                                        <h3 className="action-card-title">What To Do</h3>
                                                    </div>
                                                    <ul className="space-y-2">
                                                        {result.doList.map((item, i) => (
                                                            <li key={i} className="flex items-start gap-2">
                                                                <span className="mt-0.5 flex-shrink-0 w-4 h-4 rounded-full bg-[var(--blue-dim)] text-[var(--blue-bright)] flex items-center justify-center font-bold text-[10px]">{i + 1}</span>
                                                                <span>{item}</span>
                                                            </li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}

                                            {/* What NOT To Do */}
                                            {result.dontList.length > 0 && (
                                                <div className="action-card">
                                                    <div className="flex items-center gap-2 mb-3">
                                                        <ThumbsDown className="w-4 h-4 text-[var(--blue-bright)]" />
                                                        <h3 className="action-card-title">What NOT To Do</h3>
                                                    </div>
                                                    <ul className="space-y-2">
                                                        {result.dontList.map((item, i) => (
                                                            <li key={i} className="flex items-start gap-2">
                                                                <span className="mt-0.5 flex-shrink-0 w-4 h-4 rounded-full bg-[var(--blue-dim)] text-[var(--blue-bright)] flex items-center justify-center font-bold text-[10px]">{i + 1}</span>
                                                                <span>{item}</span>
                                                            </li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}
                                        </motion.div>
                                    )}

                                </>
                            ) : (
                                <div className="h-full flex flex-col items-center justify-center p-8 text-center border-2 border-dashed border-[var(--border-subtle)] rounded-2xl bg-[var(--bg-surface)]">
                                    <div className="w-16 h-16 bg-[var(--blue-dim)] rounded-full flex items-center justify-center mb-4">
                                        <FileText className="w-8 h-8 text-[var(--blue-bright)]" />
                                    </div>
                                    <h3 className="text-lg font-bold text-white" style={{ fontFamily: 'var(--font-display)' }}>No Analysis Yet</h3>
                                    <p className="max-w-xs mx-auto mt-2 text-[var(--white-muted)] text-sm">
                                        Upload an image to see the AI detection results and personalised do/don't recommendations here.
                                    </p>
                                </div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>
            </div>
        </div>
    );
}
