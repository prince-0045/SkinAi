import React, { useState, useEffect } from 'react';
import { Helmet } from 'react-helmet-async';
import DoctorFinder from '../components/DoctorFinder';
import { MapPin } from 'lucide-react';

export default function Doctors() {
    return (
        <div className="bg-base min-h-screen transition-colors duration-300 pt-24 pb-12">
            <Helmet>
                <title>Find Doctors - DERMAURA</title>
            </Helmet>

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 page-enter">
                <div className="text-center mb-10">
                    <div className="location-icon-badge mx-auto mb-4">
                        <MapPin className="w-6 h-6" />
                    </div>
                    <h1 className="page-title sm:text-4xl text-3xl">Find Dermatologists Nearby</h1>
                    <p className="page-subtitle mt-4 max-w-2xl mx-auto">
                        Locate verified specialists in your area. Get directions, check ratings, and book appointments directly.
                    </p>
                </div>

                <DoctorFinder />
            </div>
        </div>
    );
}
