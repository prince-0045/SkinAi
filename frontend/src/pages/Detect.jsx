import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, AlertCircle, FileText, CheckCircle } from 'lucide-react';
import ScanningLoader from '../components/animations/ScanningLoader';

export default function Detect() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [result, setResult] = useState(null);

    const onDrop = useCallback(acceptedFiles => {
        const selectedFile = acceptedFiles[0];
        if (selectedFile) {
            setFile(selectedFile);
            const objectUrl = URL.createObjectURL(selectedFile);
            setPreview(objectUrl);
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
        setIsAnalyzing(false);
    };

    const handleAnalyze = () => {
        if (!file) return;
        setIsAnalyzing(true);

        // Simulate API call
        setTimeout(() => {
            setIsAnalyzing(false);
            setResult({
                disease: "Eczema",
                confidence: 94.2,
                severity: "Moderate",
                description: "A condition that makes your skin red and itchy. It's common in children but can occur at any age.",
                recommendation: "Keep skin moisturized. Apply cool compresses. Avoid irritants."
            });
        }, 3000);
    };

    return (
        <div className="min-h-screen bg-gray-50 pt-20 pb-12">
            <div className="max-w-4xl mx-auto px-4 sm:px-6">
                <div className="text-center mb-10">
                    <h1 className="text-3xl font-bold text-gray-900">AI Skin Disease Detection</h1>
                    <p className="mt-2 text-gray-600">Upload a clear photo of the affected area for instant analysis.</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* Upload Area */}
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
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
                                <p className="text-lg font-medium text-gray-900">Click to upload or drag & drop</p>
                                <p className="text-sm text-gray-500 mt-1">SVG, PNG, JPG or GIF (max. 5MB)</p>
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

                                {/* Overlay when analyzing */}
                                {isAnalyzing && (
                                    <div className="absolute inset-0 bg-black/60 flex items-center justify-center z-10">
                                        <ScanningLoader />
                                    </div>
                                )}
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

                    {/* Results Area */}
                    <div className="space-y-6">
                        <AnimatePresence>
                            {result ? (
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="bg-white p-6 rounded-2xl shadow-lg border-t-4 border-medical-500"
                                >
                                    <div className="flex items-center justify-between mb-4">
                                        <h2 className="text-xl font-bold text-gray-900">Analysis Result</h2>
                                        <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-semibold flex items-center">
                                            <CheckCircle className="w-4 h-4 mr-1" /> Complete
                                        </span>
                                    </div>

                                    <div className="mb-6">
                                        <p className="text-sm text-gray-500 mb-1">Detected Condition</p>
                                        <div className="text-3xl font-bold text-medical-700">{result.disease}</div>
                                        <div className="mt-2 w-full bg-gray-200 rounded-full h-2.5">
                                            <div className="bg-medical-600 h-2.5 rounded-full" style={{ width: `${result.confidence}%` }}></div>
                                        </div>
                                        <div className="flex justify-between mt-1 text-xs text-gray-500">
                                            <span>Confidence Score</span>
                                            <span className="font-bold">{result.confidence}%</span>
                                        </div>
                                    </div>

                                    <div className="space-y-4">
                                        <div className="p-4 bg-medical-50 rounded-lg">
                                            <h3 className="font-semibold text-medical-900 mb-1">Description</h3>
                                            <p className="text-sm text-medical-800">{result.description}</p>
                                        </div>

                                        <div className="p-4 bg-yellow-50 rounded-lg">
                                            <h3 className="font-semibold text-yellow-900 mb-1 flex items-center">
                                                <AlertCircle className="w-4 h-4 mr-2" />
                                                Recommendation
                                            </h3>
                                            <p className="text-sm text-yellow-800">{result.recommendation}</p>
                                        </div>
                                    </div>

                                    <div className="mt-6 pt-4 border-t border-gray-100 text-center">
                                        <button className="text-medical-600 font-medium text-sm hover:underline flex items-center justify-center mx-auto">
                                            <FileText className="w-4 h-4 mr-1" /> Save to Health Record
                                        </button>
                                    </div>
                                </motion.div>
                            ) : (
                                <div className="h-full flex flex-col items-center justify-center p-8 text-center text-gray-400 bg-white rounded-2xl border border-gray-100 border-dashed">
                                    <div className="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center mb-4">
                                        <FileText className="w-8 h-8 text-gray-300" />
                                    </div>
                                    <h3 className="text-lg font-medium text-gray-900">No Analysis Yet</h3>
                                    <p className="max-w-xs mx-auto mt-2">Upload an image to see the AI detection results and recommendations here.</p>
                                </div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>
            </div>
        </div>
    );
}
