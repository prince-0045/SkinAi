import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, AlertCircle, FileText, CheckCircle, ShieldAlert, ThumbsUp, ThumbsDown, Info } from 'lucide-react';
import ScanningLoader from '../components/animations/ScanningLoader';
import { useAuth } from '../context/AuthContext';
import DoctorFinder from '../components/DoctorFinder';
import { downloadMedicalReport } from '../utils/pdfGenerator';

const SEVERITY_CONFIG = {
    Low:      { bg: 'bg-green-100',  text: 'text-green-800',  border: 'border-green-400',  dot: 'bg-green-500'  },
    Moderate: { bg: 'bg-yellow-100', text: 'text-yellow-800', border: 'border-yellow-400', dot: 'bg-yellow-500' },
    High:     { bg: 'bg-orange-100', text: 'text-orange-800', border: 'border-orange-400', dot: 'bg-orange-500' },
    Critical: { bg: 'bg-red-100',    text: 'text-red-800',    border: 'border-red-500',    dot: 'bg-red-600'    },
    Unknown:  { bg: 'bg-gray-100',   text: 'text-gray-700',   border: 'border-gray-400',   dot: 'bg-gray-400'   },
};

export default function Detect() {
    const [file, setFile]           = useState(null);
    const [preview, setPreview]     = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [result, setResult]       = useState(null);
    const [error, setError]         = useState(null);
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
                    disease:        data.disease_detected,
                    confidence:     (data.confidence_score * 100).toFixed(1),
                    severity:       data.severity_level,
                    description:    data.description  || 'AI analysis complete based on the uploaded image.',
                    recommendation: data.recommendation || 'Please consult a dermatologist for a professional diagnosis.',
                    doList:         data.do_list   || [],
                    dontList:       data.dont_list || [],
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
        <div className="bg-[#0d1117] bg-navy-mesh min-h-screen transition-colors duration-300 pt-20 pb-12">
            <div className="max-w-5xl mx-auto px-4 sm:px-6">
                <div className="text-center mb-10">
                    <h1 className="text-3xl font-bold text-gray-900 dark:text-white">AI Skin Disease Detection</h1>
                    <p className="mt-2 text-gray-600 dark:text-gray-400">Upload a clear photo of the affected area for instant analysis.</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* ── Upload Area ── */}
                    <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-slate-700 transition-colors duration-300">
                        {!preview ? (
                            <div
                                {...getRootProps()}
                                className={`border-2 border-dashed rounded-xl h-80 flex flex-col items-center justify-center cursor-pointer transition-colors
                                    ${isDragActive ? 'border-medical-500 bg-medical-50' : 'border-gray-300 hover:border-medical-400'}`}
                            >
                                <input {...getInputProps()} />
                                <div className="w-16 h-16 bg-medical-50 rounded-full flex items-center justify-center mb-4 text-medical-600">
                                    <Upload className="w-8 h-8" />
                                </div>
                                <p className="text-lg font-medium text-gray-900 dark:text-gray-200">Click to upload or drag &amp; drop</p>
                                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">SVG, PNG, JPG or GIF (max. 5MB)</p>
                                {uploadLimit && (
                                    <div className="mt-3 inline-flex items-center gap-2 px-3 py-1.5 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-lg text-sm font-medium border border-blue-100 dark:border-blue-800/50">
                                        <Info className="w-4 h-4" />
                                        <span>{uploadLimit.remaining} of {uploadLimit.limit} daily uploads remaining (resets in 24h)</span>
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
                            <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl flex items-start gap-3">
                                <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                                <div>
                                    <h3 className="text-sm font-semibold text-red-900 dark:text-red-300">Upload Limit Reached</h3>
                                    <p className="text-sm text-red-700 dark:text-red-200 mt-1">{error}</p>
                                </div>
                            </div>
                        )}

                        {preview && !isAnalyzing && !result && (
                            <button
                                onClick={handleAnalyze}
                                className="w-full mt-4 py-3 bg-medical-600 text-white rounded-lg font-medium hover:bg-medical-700 transition shadow-lg shadow-medical-500/30"
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
                                        className={`bg-white dark:bg-slate-800 p-6 rounded-2xl shadow-lg border-t-4 ${sevCfg.border}`}
                                    >
                                        {/* Header */}
                                        <div className="flex items-center justify-between mb-4">
                                            <h2 className="text-xl font-bold text-gray-900 dark:text-white">Analysis Result</h2>
                                            <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-semibold flex items-center gap-1">
                                                <CheckCircle className="w-4 h-4" /> Complete
                                            </span>
                                        </div>

                                        {/* Disease + Confidence */}
                                        <div className="mb-5">
                                            <p className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-1">Detected Condition</p>
                                            <div className="text-2xl font-bold text-medical-700 dark:text-medical-400">{result.disease}</div>

                                            {/* Severity badge */}
                                            <span className={`inline-flex items-center gap-1.5 mt-2 px-3 py-1 rounded-full text-xs font-semibold ${sevCfg.bg} ${sevCfg.text}`}>
                                                <span className={`w-2 h-2 rounded-full ${sevCfg.dot}`}></span>
                                                {result.severity} Severity
                                            </span>

                                            {/* Confidence bar */}
                                            <div className="mt-3">
                                                <div className="w-full bg-gray-200 dark:bg-slate-600 rounded-full h-2.5">
                                                    <div
                                                        className="bg-medical-600 h-2.5 rounded-full transition-all duration-1000"
                                                        style={{ width: `${result.confidence}%` }}
                                                    />
                                                </div>
                                                <div className="flex justify-between mt-1 text-xs text-gray-500 dark:text-gray-400">
                                                    <span>Confidence Score</span>
                                                    <span className="font-bold">{result.confidence}%</span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Description */}
                                        <div className="p-4 bg-medical-50 dark:bg-medical-900/20 rounded-xl border border-medical-100 dark:border-medical-800 mb-4">
                                            <h3 className="font-semibold text-medical-900 dark:text-medical-300 flex items-center gap-2 mb-2">
                                                <Info className="w-4 h-4" /> About This Condition
                                            </h3>
                                            <p className="text-sm text-medical-800 dark:text-medical-200 leading-relaxed">{result.description}</p>
                                        </div>

                                        {/* Recommendation */}
                                        <div className={`p-4 rounded-xl border ${result.severity === 'Critical' ? 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800' : 'bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-800'}`}>
                                            <h3 className={`font-semibold flex items-center gap-2 mb-1 text-sm ${result.severity === 'Critical' ? 'text-red-900 dark:text-red-300' : 'text-yellow-900 dark:text-yellow-300'}`}>
                                                <AlertCircle className="w-4 h-4" /> Doctor's Advice
                                            </h3>
                                            <p className={`text-sm ${result.severity === 'Critical' ? 'text-red-800 dark:text-red-200 font-semibold' : 'text-yellow-800 dark:text-yellow-200'}`}>
                                                {result.recommendation}
                                            </p>
                                        </div>

                                        {/* Save button */}
                                        <div className="mt-5 pt-4 border-t border-gray-100 dark:border-slate-700 text-center">
                                            <button 
                                                onClick={handleDownloadPDF}
                                                className="text-medical-600 font-medium text-sm hover:underline flex items-center justify-center mx-auto gap-1"
                                            >
                                                <FileText className="w-4 h-4" /> Save to Health Record
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
                                                <div className="bg-white dark:bg-slate-800 rounded-2xl border border-green-200 dark:border-green-900 shadow-sm overflow-hidden">
                                                    <div className="bg-green-500 px-4 py-3 flex items-center gap-2">
                                                        <ThumbsUp className="w-4 h-4 text-white" />
                                                        <h3 className="text-sm font-bold text-white">What To Do</h3>
                                                    </div>
                                                    <ul className="p-4 space-y-2">
                                                        {result.doList.map((item, i) => (
                                                            <li key={i} className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300">
                                                                <span className="mt-0.5 flex-shrink-0 w-5 h-5 bg-green-100 dark:bg-green-900/40 text-green-600 dark:text-green-400 rounded-full flex items-center justify-center font-bold text-xs">{i + 1}</span>
                                                                <span>{item}</span>
                                                            </li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}

                                            {/* What NOT To Do */}
                                            {result.dontList.length > 0 && (
                                                <div className="bg-white dark:bg-slate-800 rounded-2xl border border-red-200 dark:border-red-900 shadow-sm overflow-hidden">
                                                    <div className="bg-red-500 px-4 py-3 flex items-center gap-2">
                                                        <ThumbsDown className="w-4 h-4 text-white" />
                                                        <h3 className="text-sm font-bold text-white">What NOT To Do</h3>
                                                    </div>
                                                    <ul className="p-4 space-y-2">
                                                        {result.dontList.map((item, i) => (
                                                            <li key={i} className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300">
                                                                <span className="mt-0.5 flex-shrink-0 w-5 h-5 bg-red-100 dark:bg-red-900/40 text-red-600 dark:text-red-400 rounded-full flex items-center justify-center font-bold text-xs">{i + 1}</span>
                                                                <span>{item}</span>
                                                            </li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}
                                        </motion.div>
                                    )}

                                    {/* ── Doctor Finder ── */}
                                    <motion.div
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: 0.3 }}
                                    >
                                        <DoctorFinder />
                                    </motion.div>
                                </>
                            ) : (
                                <div className="h-full flex flex-col items-center justify-center p-8 text-center text-gray-400 bg-white dark:bg-slate-800 rounded-2xl border border-gray-100 dark:border-slate-700 border-dashed">
                                    <div className="w-16 h-16 bg-gray-50 dark:bg-slate-700 rounded-full flex items-center justify-center mb-4">
                                        <FileText className="w-8 h-8 text-gray-300 dark:text-gray-500" />
                                    </div>
                                    <h3 className="text-lg font-medium text-gray-900 dark:text-white">No Analysis Yet</h3>
                                    <p className="max-w-xs mx-auto mt-2 text-gray-500 dark:text-gray-400">
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
