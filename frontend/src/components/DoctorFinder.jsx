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
        <div className="panel card">
            <div className="p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                    <h3 className="text-lg font-bold text-[var(--white-full)] flex items-center" style={{ fontFamily: 'var(--font-display)' }}>
                        <MapPin className="w-5 h-5 text-[var(--blue-bright)] mr-2" />
                        Find Dermatologists Nearby
                    </h3>
                    <p className="text-sm text-[var(--white-muted)] mt-1">
                        Use AWS Location Service to locate professionals directly, or open Google Maps.
                    </p>
                </div>
                <div className="flex flex-col sm:flex-row gap-3">
                    <button
                        onClick={findAwsDoctors}
                        disabled={loading}
                        className="btn-primary flex items-center justify-center disabled:opacity-50"
                    >
                        {loading ? <Loader className="w-4 h-4 mr-2 animate-spin" /> : <Navigation className="w-4 h-4 mr-2" />}
                        {loading ? 'Searching AWS...' : 'Find Near Me'}
                    </button>
                    <button
                        onClick={openGoogleMaps}
                        className="btn-secondary flex items-center justify-center"
                        title="Open directly in Google Maps"
                    >
                        <MapIcon className="w-4 h-4 mr-2" />
                        Google Maps
                    </button>
                </div>
            </div>

            {error && (
                <div className="px-6 pb-4">
                    <div className="p-3 bg-[var(--blue-dim)] text-[var(--white-soft)] text-sm rounded-lg border border-[var(--border-subtle)] flex justify-between items-center sm:flex-row flex-col gap-2">
                        <span className="text-center sm:text-left">{error}</span>
                        <button
                            onClick={openGoogleMaps}
                            className="text-[var(--blue-bright)] font-semibold underline text-xs whitespace-nowrap"
                        >
                            Or open Google Maps directly
                        </button>
                    </div>
                </div>
            )}

            {/* Inline AWS Doctors Results */}
            {hasSearched && !error && (
                <div className="px-6 pb-6 pt-4 border-t border-[var(--border-subtle)] bg-[var(--bg-surface)]">
                    <h4 className="text-sm font-semibold text-[var(--white-muted)] mb-4 uppercase tracking-wider flex items-center">
                        <img src="https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg" alt="AWS" className="h-4 mr-2 object-contain brightness-0 dark:invert opacity-70" />
                        Search Results from AWS Location Service
                    </h4>
                    {doctors.length > 0 ? (
                        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                            {doctors.map((doc, idx) => (
                                <div key={idx} className="doctor-card">
                                    <h3>{doc.name}</h3>
                                    <p className="text-sm text-[var(--white-muted)] mt-2 flex items-start leading-relaxed">
                                        <MapPin className="w-3.5 h-3.5 mr-1.5 mt-1 flex-shrink-0 opacity-60" />
                                        {doc.address || 'Address not listed'}
                                    </p>

                                    <div className="mt-4 pt-3 border-t border-[var(--border-subtle)]">
                                        <a
                                            href={`https://www.google.com/maps/search/?api=1&query=${doc.geometry?.location?.lat},${doc.geometry?.location?.lng}`}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="view-directions-link"
                                        >
                                            View Directions <ExternalLink className="w-3 h-3" />
                                        </a>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-center py-8 rounded-xl border border-[var(--border-subtle)] shadow-sm bg-[var(--bg-card)]">
                            <p className="text-[var(--white-muted)] font-medium">No dermatologists found nearby via AWS.</p>
                            <button onClick={openGoogleMaps} className="btn-primary mt-3 flex items-center mx-auto text-sm">
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
