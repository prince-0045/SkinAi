import React from 'react';
import { Activity, Heart } from 'lucide-react';

export default function Footer() {
    return (
        <footer className="bg-slate-900 text-slate-300 py-12">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
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

                    <div>
                        <h3 className="text-white font-semibold mb-4">Platform</h3>
                        <ul className="space-y-2 text-sm">
                            <li><a href="#" className="hover:text-medical-400 transition-colors">Detection</a></li>
                            <li><a href="#" className="hover:text-medical-400 transition-colors">Tracker</a></li>
                            <li><a href="#" className="hover:text-medical-400 transition-colors">Pricing</a></li>
                        </ul>
                    </div>

                    <div>
                        <h3 className="text-white font-semibold mb-4">Legal</h3>
                        <ul className="space-y-2 text-sm">
                            <li><a href="#" className="hover:text-medical-400 transition-colors">Privacy Policy</a></li>
                            <li><a href="#" className="hover:text-medical-400 transition-colors">Terms of Service</a></li>
                            <li><a href="#" className="hover:text-medical-400 transition-colors">Medical Disclaimer</a></li>
                        </ul>
                    </div>

                    <div>
                        <h3 className="text-white font-semibold mb-4">Contact</h3>
                        <ul className="space-y-2 text-sm">
                            <li>help@skinai.com</li>
                            <li>1-800-SKIN-AI</li>
                        </ul>
                    </div>
                </div>

                <div className="border-t border-slate-800 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center text-sm">
                    <p>&copy; {new Date().getFullYear()} SkinAi. All rights reserved.</p>
                    <p className="flex items-center mt-4 md:mt-0">
                        Made with <Heart className="h-4 w-4 text-red-500 mx-1 fill-current" /> for healthy skin
                    </p>
                </div>
            </div>
        </footer>
    );
}
