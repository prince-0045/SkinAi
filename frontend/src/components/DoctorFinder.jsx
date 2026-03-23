import React, { useState } from 'react';
import { MapPin, Navigation, ExternalLink, Loader, Map as MapIcon, Star } from 'lucide-react';

export default function DoctorFinder() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [doctors, setDoctors] = useState([]);
    const [hasSearched, setHasSearched] = useState(false);

    // Direct Google Maps redirect
    const openGoogleMaps = () => {
        window.open('https://www.google.com/maps/search/dermatologist+near+me', '_blank', 'noopener,noreferrer');
    };

    // Fetch doctors from Python Backend (AWS Location APIs)
    const findAwsDoctors = async () => {
        setLoading(true);
        setError(null);
        setHasSearched(false);
        try {
            const position = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject, {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 300000
                });
            });
            
            const { latitude, longitude } = position.coords;
            const API_BASE_URL = import.meta.env.VITE_API_URL || '';
            
            const response = await fetch(`${API_BASE_URL}/api/v1/doctors/nearby`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ latitude, longitude })
            });

            if (!response.ok) {
                throw new Error('Failed to fetch dermatologists from the server.');
            }
            
            const data = await response.json();
            setDoctors(data);
            setHasSearched(true);
        } catch (err) {
            setError(err.message || 'Could not access your location or connect to the server.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-gray-100 dark:border-slate-700 overflow-hidden transition-colors duration-300">
            <div className="p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white flex items-center">
                        <MapPin className="w-5 h-5 text-medical-600 dark:text-medical-400 mr-2" />
                        Find Dermatologists Nearby
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                        Use AWS Location Service to locate professionals directly, or open Google Maps.
                    </p>
                </div>
                <div className="flex flex-col sm:flex-row gap-3">
                    <button
                        onClick={findAwsDoctors}
                        disabled={loading}
                        className="flex-1 sm:flex-none justify-center bg-medical-600 hover:bg-medical-700 text-white px-5 py-2.5 rounded-lg font-medium transition-all shadow-sm flex items-center disabled:opacity-50 whitespace-nowrap"
                    >
                        {loading ? <Loader className="w-4 h-4 mr-2 animate-spin" /> : <Navigation className="w-4 h-4 mr-2" />}
                        {loading ? 'Searching AWS...' : 'Find Near Me'}
                    </button>
                    <button
                        onClick={openGoogleMaps}
                        className="flex-1 sm:flex-none justify-center px-5 py-2.5 border-2 border-gray-200 dark:border-slate-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-slate-700 transition flex items-center font-medium whitespace-nowrap"
                        title="Open directly in Google Maps"
                    >
                        <MapIcon className="w-4 h-4 mr-2" />
                        Google Maps
                    </button>
                </div>
            </div>

            {error && (
                <div className="px-6 pb-4">
                    <div className="p-3 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 text-sm rounded-lg border border-red-200 dark:border-red-800 flex justify-between items-center sm:flex-row flex-col gap-2">
                        <span className="text-center sm:text-left">{error}</span>
                        <button
                            onClick={openGoogleMaps}
                            className="text-red-600 dark:text-red-300 font-semibold underline text-xs whitespace-nowrap"
                        >
                            Or open Google Maps directly
                        </button>
                    </div>
                </div>
            )}

            {/* Inline AWS Doctors Results */}
            {hasSearched && !error && (
                <div className="px-6 pb-6 pt-4 border-t border-gray-100 dark:border-slate-700 bg-gray-50/50 dark:bg-slate-800/30">
                    <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4 uppercase tracking-wider flex items-center">
                        <img src="https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg" alt="AWS" className="h-4 mr-2 object-contain dark:brightness-0 dark:invert" />
                        Search Results from AWS Location Service
                    </h4>
                    {doctors.length > 0 ? (
                        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                            {doctors.map((doc, idx) => (
                                <div key={idx} className="p-4 rounded-xl border border-gray-200 dark:border-slate-600 bg-white dark:bg-slate-800 hover:shadow-md hover:border-medical-300 dark:hover:border-medical-700 transition-all group">
                                    <h5 className="font-semibold text-gray-900 dark:text-white text-[15px] leading-snug group-hover:text-medical-600 dark:group-hover:text-medical-400 transition-colors">{doc.name}</h5>
                                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-2 flex items-start leading-relaxed">
                                        <MapPin className="w-3.5 h-3.5 mr-1.5 mt-1 flex-shrink-0 text-gray-400" />
                                        {doc.address || 'Address not listed'}
                                    </p>
                                    
                                    <div className="mt-4 pt-3 border-t border-gray-100 dark:border-slate-700">
                                        <a
                                            href={`https://www.google.com/maps/search/?api=1&query=${doc.geometry?.location?.lat},${doc.geometry?.location?.lng}`}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-medical-600 dark:text-medical-400 text-sm font-medium hover:underline flex items-center w-max"
                                        >
                                            View Directions <ExternalLink className="w-3 h-3 ml-1.5" />
                                        </a>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-center py-8 bg-white dark:bg-slate-800 rounded-xl border border-gray-200 dark:border-slate-600 shadow-sm">
                            <p className="text-gray-500 dark:text-gray-400 font-medium">No dermatologists found nearby via AWS.</p>
                            <button onClick={openGoogleMaps} className="mt-3 text-white bg-medical-600 hover:bg-medical-700 px-4 py-2 rounded-md font-medium text-sm transition-colors mx-auto flex items-center">
                                <MapIcon className="w-3.5 h-3.5 mr-1.5" />
                                Search broadly on Google Maps
                            </button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
