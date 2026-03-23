import React from 'react';
import { Helmet } from 'react-helmet-async';
import { Shield, Database, Eye, Trash2, Mail, Clock } from 'lucide-react';

const Section = ({ icon: Icon, title, children }) => (
    <div className="mb-8">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2 mb-4">
            <Icon className="w-5 h-5 text-medical-600" />
            {title}
        </h2>
        <div className="text-gray-600 dark:text-gray-400 space-y-3 leading-relaxed">
            {children}
        </div>
    </div>
);

export default function Privacy() {
    return (
        <div className="bg-[#0d1117] bg-navy-mesh min-h-screen transition-colors duration-300 pt-24 pb-12">
            <Helmet>
                <title>Privacy Policy - DERMAURA</title>
                <meta name="description" content="DERMAURA privacy policy — how we handle your data." />
            </Helmet>

            <div className="max-w-3xl mx-auto px-4 sm:px-6">
                <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-gray-100 dark:border-slate-700 p-8 md:p-12">
                    <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Privacy Policy</h1>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-8">Last updated: {new Date().toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}</p>

                    <Section icon={Shield} title="Overview">
                        <p>
                            DERMAURA is an AI-powered skin analysis tool designed for educational and informational purposes.
                            We are committed to protecting your privacy and handling your data responsibly.
                        </p>
                        <p className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-3 text-amber-800 dark:text-amber-200 text-sm">
                            <strong>Important:</strong> DERMAURA is NOT a medical device and does NOT provide medical diagnoses.
                            Always consult a qualified healthcare professional for medical advice.
                        </p>
                    </Section>

                    <Section icon={Database} title="Data We Collect">
                        <ul className="list-disc list-inside space-y-2">
                            <li><strong>Account information:</strong> Name, email address (for authentication)</li>
                            <li><strong>Uploaded images:</strong> Skin photos you submit for analysis</li>
                            <li><strong>Analysis results:</strong> AI-generated predictions and confidence scores</li>
                            <li><strong>Usage data:</strong> Scan timestamps and basic interaction logs</li>
                        </ul>
                    </Section>

                    <Section icon={Eye} title="How We Use Your Data">
                        <ul className="list-disc list-inside space-y-2">
                            <li>To provide AI-assisted skin analysis</li>
                            <li>To maintain your scan history for personal tracking</li>
                            <li>To authenticate your account</li>
                            <li>To generate PDF reports at your request</li>
                        </ul>
                        <p>We do <strong>not</strong> sell, share, or use your images or data for advertising purposes.</p>
                    </Section>

                    <Section icon={Shield} title="Data Storage & Security">
                        <ul className="list-disc list-inside space-y-2">
                            <li>Images are stored securely on Cloudinary (encrypted in transit and at rest)</li>
                            <li>User data is stored in MongoDB Atlas with TLS encryption</li>
                            <li>Passwords are hashed using Argon2 (industry-standard)</li>
                            <li>Authentication uses JWT tokens with expiration</li>
                        </ul>
                    </Section>

                    <Section icon={Trash2} title="Data Deletion">
                        <p>
                            You can delete your account and all associated data (including uploaded images and scan history)
                            at any time from your Profile page. Once deleted, your data cannot be recovered.
                        </p>
                    </Section>

                    <Section icon={Clock} title="Data Retention">
                        <p>
                            We retain your data for as long as your account is active. If you delete your account,
                            all personal data is permanently removed from our systems.
                        </p>
                    </Section>

                    <Section icon={Mail} title="Contact">
                        <p>
                            If you have questions about this privacy policy or your data, please contact us
                            through the application or at the email address provided in your account settings.
                        </p>
                    </Section>
                </div>
            </div>
        </div>
    );
}
