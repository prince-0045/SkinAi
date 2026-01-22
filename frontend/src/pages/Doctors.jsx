import React, { useState, useEffect } from 'react';
import { Helmet } from 'react-helmet-async';
import DoctorFinder from '../components/DoctorFinder';
import { MapPin } from 'lucide-react';

export default function Doctors() {
    return (
        <div className="min-h-screen bg-gray-50 dark:bg-slate-900 transition-colors duration-300 pt-24 pb-12">
            <Helmet>
                <title>Find Doctors - SkinAi</title>
            </Helmet>

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="text-center mb-10">
                    <div className="inline-flex items-center justify-center p-3 bg-medical-100 dark:bg-medical-900/30 rounded-full mb-4">
                        <MapPin className="w-8 h-8 text-medical-600 dark:text-medical-400" />
                    </div>
                    <h1 className="text-3xl font-bold text-gray-900 dark:text-white sm:text-4xl">Find Dermatologists Nearby</h1>
                    <p className="mt-4 text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
                        Locate verified specialists in your area. Get directions, check ratings, and book appointments directly.
                    </p>
                </div>

                <DoctorFinder />
            </div>
        </div>
    );
}
