import React, { useState } from 'react';
import { MapPin, Star, Navigation, Phone, ExternalLink, Loader } from 'lucide-react';

export default function DoctorFinder({ initialCoords }) {
    const [doctors, setDoctors] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [searched, setSearched] = useState(false);

    const findDoctors = async () => {
        setLoading(true);
        setError(null);
        try {
            // Use browser geolocation if initialCoords not provided
            let lat, lng;

            if (initialCoords) {
                lat = initialCoords.latitude;
                lng = initialCoords.longitude;
            } else {
                const position = await new Promise((resolve, reject) => {
                    navigator.geolocation.getCurrentPosition(resolve, reject);
                });
                lat = position.coords.latitude;
                lng = position.coords.longitude;
            }

            const response = await fetch('http://localhost:8000/api/v1/doctors/nearby', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({ latitude: lat, longitude: lng })
            });

            if (!response.ok) throw new Error('Failed to fetch doctors');

            const data = await response.json();
            setDoctors(data);
            setSearched(true);
        } catch (err) {
            console.error(err);
            setError("Could not find doctors nearby. Please allow location access.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-gray-100 dark:border-slate-700 overflow-hidden mt-8 transition-colors duration-300">
            <div className="p-6 border-b border-gray-100 dark:border-slate-700 flex justify-between items-center bg-gray-50 dark:bg-slate-900/50">
                <div>
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white flex items-center">
                        <MapPin className="w-5 h-5 text-medical-600 mr-2" />
                        Find Dermatologists Nearby
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Locate specialists in your area based on your current location.</p>
                </div>
                {!searched && (
                    <button
                        onClick={findDoctors}
                        disabled={loading}
                        className="bg-medical-600 hover:bg-medical-700 text-white px-4 py-2 rounded-lg font-medium transition flex items-center disabled:opacity-50"
                    >
                        {loading ? <Loader className="w-4 h-4 mr-2 animate-spin" /> : <Navigation className="w-4 h-4 mr-2" />}
                        {loading ? 'Searching...' : 'Find Near Me'}
                    </button>
                )}
            </div>

            {error && (
                <div className="p-4 bg-red-50 text-red-700 text-sm border-b border-red-100">
                    {error}
                </div>
            )}

            {searched && (
                <div className="divide-y divide-gray-100">
                    {doctors.length === 0 ? (
                        <div className="p-8 text-center text-gray-500">
                            No dermatologists found nearby.
                        </div>
                    ) : (
                        doctors.map((doc, index) => (
                            <div key={index} className="p-4 hover:bg-gray-50 transition flex flex-col sm:flex-row sm:items-center justify-between">
                                <div className="mb-4 sm:mb-0">
                                    <h4 className="font-semibold text-gray-900 dark:text-white">{doc.name}</h4>
                                    <div className="flex items-center text-sm text-gray-600 dark:text-gray-400 mt-1">
                                        <MapPin className="w-3 h-3 mr-1" />
                                        <span>{doc.address}</span>
                                    </div>
                                    <div className="flex items-center text-sm mt-2">
                                        {doc.rating && (
                                            <div className="flex items-center bg-yellow-50 text-yellow-700 px-2 py-0.5 rounded mr-3">
                                                <Star className="w-3 h-3 fill-current mr-1" />
                                                <span className="font-medium">{doc.rating}</span>
                                                <span className="text-xs text-yellow-600 ml-1">({doc.user_ratings_total})</span>
                                            </div>
                                        )}
                                        {doc.open_now !== null && doc.open_now !== undefined && (
                                            <span className={`text-xs font-medium px-2 py-0.5 rounded ${doc.open_now ? 'text-green-600 bg-green-50' : 'text-red-600 bg-red-50'}`}>
                                                {doc.open_now ? 'Open Now' : 'Closed'}
                                            </span>
                                        )}
                                    </div>
                                </div>
                                <div className="flex space-x-3">
                                    <a
                                        href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(doc.name + ' ' + doc.address)}&query_place_id=${doc.place_id}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="flex items-center px-3 py-2 border border-gray-200 dark:border-slate-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-white dark:hover:bg-slate-700 hover:border-medical-300 hover:text-medical-600 transition text-sm font-medium"
                                    >
                                        <ExternalLink className="w-4 h-4 mr-2" />
                                        Directions
                                    </a>
                                </div>
                            </div>
                        ))
                    )}

                    <div className="p-4 bg-gray-50 text-center border-t border-gray-100">
                        <button
                            onClick={findDoctors}
                            disabled={loading}
                            className="text-medical-600 text-sm font-medium hover:underline"
                        >
                            Refresh Location
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
