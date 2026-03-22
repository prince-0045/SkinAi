import React from 'react';
import { Activity, Heart, Shield } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Footer() {
    return (
        <footer className="bg-slate-900 text-slate-300">
            {/* Medical Disclaimer */}
            <div className="bg-amber-950/50 border-t border-amber-900/30 py-3 px-4 text-center">
                <p className="text-xs text-amber-200/80 flex items-center justify-center gap-1.5 flex-wrap">
                    <Shield className="w-3.5 h-3.5 flex-shrink-0" />
                    <span>
                        <strong>AI Disclaimer:</strong> SkinAi provides AI-assisted analysis only — this is{' '}
                        <strong>NOT a medical diagnosis</strong>. Always consult a licensed dermatologist for professional advice.
                    </span>
                </p>
            </div>

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                <div className="flex flex-col items-center text-center">
                    <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                            <div className="p-1.5 bg-medical-500 rounded-lg">
                                <Activity className="h-5 w-5 text-white" />
                            </div>
                            <span className="text-lg font-bold text-white">SkinAi</span>
                        </div>
                        <p className="text-sm text-slate-400">
                            Advanced skin disease detection and healing tracking powered by AI.
                        </p>
                    </div>
                </div>

                <div className="border-t border-slate-800 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center text-sm gap-4">
                    <p>&copy; {new Date().getFullYear()} SkinAi. All rights reserved.</p>
                    <div className="flex items-center gap-6">
                        <Link to="/privacy" className="text-slate-400 hover:text-white transition">Privacy Policy</Link>
                        <p className="flex items-center">
                            Made with <Heart className="h-4 w-4 text-red-500 mx-1 fill-current" /> for healthy skin
                        </p>
                    </div>
                </div>
            </div>
        </footer>
    );
}
