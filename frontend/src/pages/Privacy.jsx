import React from 'react';
import { Helmet } from 'react-helmet-async';
import { Shield, Database, Eye, Trash2, Mail, Clock } from 'lucide-react';

const Section = ({ icon: Icon, title, children }) => (
    <div className="mb-8">
        <h2 className="text-xl font-bold text-white flex items-center gap-2 mb-4" style={{ fontFamily: 'var(--font-display)' }}>
            <Icon className="w-5 h-5 text-[var(--blue-bright)]" />
            {title}
        </h2>
        <div className="text-[var(--white-muted)] space-y-3 leading-relaxed">
            {children}
        </div>
    </div>
);

export default function Privacy() {
    return (
        <div className="bg-base min-h-screen transition-colors duration-300 pt-24 pb-12">
            <Helmet>
                <title>Privacy Policy - DERMAURA</title>
                <meta name="description" content="DERMAURA privacy policy — how we handle your data." />
            </Helmet>

            <div className="max-w-3xl mx-auto px-4 sm:px-6">
                <div className="card p-8 md:p-12">
                    <h1 className="text-3xl font-bold text-white mb-2" style={{ fontFamily: 'var(--font-display)' }}>Privacy Policy</h1>
                    <p className="text-sm text-[var(--white-muted)] mb-8">Last updated: {new Date().toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}</p>

                    <Section icon={Shield} title="Overview">
                        <p>
                            DERMAURA is an AI-powered skin analysis tool designed for educational and informational purposes.
                            We are committed to protecting your privacy and handling your data responsibly.
                        </p>
                        <p className="bg-[var(--blue-dim)] border border-[var(--border-subtle)] rounded-lg p-3 text-white text-sm">
                            <strong className="text-[var(--blue-bright)]">Important:</strong> DERMAURA is NOT a medical device and does NOT provide medical diagnoses.
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
