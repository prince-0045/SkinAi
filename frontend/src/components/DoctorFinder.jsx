import React, { useState } from 'react';
import { MapPin, Navigation, ExternalLink, Loader } from 'lucide-react';

export default function DoctorFinder() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const openGoogleMaps = async () => {
        setLoading(true);
        setError(null);
        try {
            const position = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject, {
                    enableHighAccuracy: false,
                    timeout: 5000,
                    maximumAge: 300000
                });
            });
            const { latitude, longitude } = position.coords;
            const url = `https://www.google.com/maps/search/dermatologist/@${latitude},${longitude},14z`;
            window.open(url, '_blank', 'noopener,noreferrer');
        } catch (err) {
            // Geolocation blocked (HTTP / denied / timeout) — fall back to Google's own location
            window.open('https://www.google.com/maps/search/dermatologist+near+me', '_blank', 'noopener,noreferrer');
        } finally {
            setLoading(false);
        }
    };

    // Fallback: open without location
    const openWithoutLocation = () => {
        window.open('https://www.google.com/maps/search/dermatologist+near+me', '_blank', 'noopener,noreferrer');
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
                        Opens Google Maps to show dermatologists near your location.
                    </p>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={openGoogleMaps}
                        disabled={loading}
                        className="bg-medical-600 hover:bg-medical-700 text-white px-4 py-2.5 rounded-lg font-medium transition flex items-center disabled:opacity-50 whitespace-nowrap"
                    >
                        {loading ? <Loader className="w-4 h-4 mr-2 animate-spin" /> : <Navigation className="w-4 h-4 mr-2" />}
                        {loading ? 'Getting Location...' : 'Find Near Me'}
                    </button>
                    <button
                        onClick={openWithoutLocation}
                        className="px-3 py-2.5 border border-gray-200 dark:border-slate-600 text-gray-600 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-slate-700 transition flex items-center text-sm"
                        title="Search without location"
                    >
                        <ExternalLink className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {error && (
                <div className="px-6 pb-4">
                    <div className="p-3 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 text-sm rounded-lg border border-red-200 dark:border-red-800 flex justify-between items-center">
                        <span>{error}</span>
                        <button
                            onClick={openWithoutLocation}
                            className="text-red-600 dark:text-red-300 underline text-xs ml-3 whitespace-nowrap"
                        >
                            Search anyway →
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
