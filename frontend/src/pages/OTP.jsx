import React, { useState, useRef, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import AuthLayout from '../components/auth/AuthLayout';
import { useAuth } from '../context/AuthContext';
import { ShieldCheck } from 'lucide-react';

export default function OTP() {
    const [otp, setOtp] = useState(['', '', '', '', '', '']);
    const inputRefs = useRef([]);
    const location = useLocation();
    const navigate = useNavigate();
    const { verifyOtp } = useAuth(); // Removed 'login' as it was dummy
    const email = location.state?.email || 'user@example.com';
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (inputRefs.current[0]) {
            inputRefs.current[0].focus();
        }
    }, []);

    const handleChange = (element, index) => {
        if (isNaN(element.value)) return;

        setOtp([...otp.map((d, idx) => (idx === index ? element.value : d))]);

        // Focus next input
        if (element.value !== '' && index < 5) {
            inputRefs.current[index + 1].focus();
        }
    };

    const handleKeyDown = (e, index) => {
        if (e.key === 'Backspace' && otp[index] === '' && index > 0) {
            inputRefs.current[index - 1].focus();
        }
    };

    const handleVerify = async (e) => {
        e.preventDefault();
        setError('');
        const code = otp.join('');

        setLoading(true);
        const res = await verifyOtp(email, code);
        setLoading(false);

        if (res.success) {
            navigate('/success');
        } else {
            setError(res.error);
        }
    };

    return (
        <AuthLayout
            title="Verify Your Email"
            subtitle={`We sent a 6-digit code to ${email}`}
        >
            <form onSubmit={handleVerify} className="space-y-8">
                <div className="flex justify-center gap-2 sm:gap-4">
                    {otp.map((data, index) => (
                        <input
                            key={index}
                            ref={el => inputRefs.current[index] = el}
                            type="text"
                            name="otp"
                            maxLength="1"
                            value={data}
                            onChange={e => handleChange(e.target, index)}
                            onKeyDown={e => handleKeyDown(e, index)}
                            className="w-10 h-12 sm:w-12 sm:h-14 border border-gray-300 rounded-lg text-center text-xl font-bold text-gray-900 focus:ring-2 focus:ring-medical-500 focus:border-medical-500 transition-all"
                        />
                    ))}
                </div>

                {error && (
                    <div className="text-red-500 text-sm text-center bg-red-50 p-2 rounded">
                        {error}
                    </div>
                )}

                <button
                    type="submit"
                    disabled={loading}
                    className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-medical-600 hover:bg-medical-700 focus:outline-none transition-colors disabled:opacity-50"
                >
                    {loading ? 'Verifying...' : 'Verify & Continue'}
                </button>

                <div className="text-center text-sm">
                    <p className="text-gray-500">Didn't receive the code?</p>
                    <button type="button" className="mt-2 text-medical-600 font-semibold hover:text-medical-500">
                        Resend Code
                    </button>
                </div>
            </form>
        </AuthLayout>
    );
}
