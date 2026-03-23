import React from 'react';

export default function DermAuraLogo({ className = '' }) {
  return (
    <svg className={className} width="680" height="200" viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="ibg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#1e4a8a"/>
          <stop offset="100%" stopColor="#0a1e42"/>
        </linearGradient>
        <linearGradient id="pg1" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stopColor="#378ADD" stopOpacity="0"/>
          <stop offset="20%"  stopColor="#378ADD"/>
          <stop offset="65%"  stopColor="#5DCAA5"/>
          <stop offset="100%" stopColor="#5DCAA5" stopOpacity="0"/>
        </linearGradient>
        <linearGradient id="pg2" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stopColor="#85B7EB" stopOpacity="0"/>
          <stop offset="45%"  stopColor="#85B7EB" stopOpacity="0.6"/>
          <stop offset="100%" stopColor="#85B7EB" stopOpacity="0"/>
        </linearGradient>
      </defs>

      {/* Outer breathing ring */}
      <circle className="rb" cx="100" cy="100" r="80" fill="none" stroke="#378ADD" strokeWidth="1" strokeDasharray="4 6"/>

      {/* Main filled circle */}
      <circle cx="100" cy="100" r="66" fill="url(#ibg)" stroke="#60a5e8" strokeWidth="2.2"/>

      {/* Inner ring accent */}
      <circle cx="100" cy="100" r="58" fill="none" stroke="#85B7EB" strokeWidth="0.6" opacity="0.3"/>

      {/* Animated pulse line */}
      <path className="pl"
        d="M37 100 L53 100 L62 100 L70 76 L80 124 L88 92 L95 100 L100 100 L105 100 L112 76 L122 124 L130 92 L138 100 L163 100"
        fill="none" stroke="url(#pg1)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>

      {/* Echo line */}
      <path className="pl2"
        d="M37 110 L53 110 L62 110 L70 90 L80 130 L88 108 L95 110 L100 110 L105 110 L112 90 L122 130 L130 108 L138 110 L163 110"
        fill="none" stroke="url(#pg2)" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>

      {/* Beat dot */}
      <circle className="bt"  cx="100" cy="100" r="4.5" fill="#5DCAA5"/>
      <circle className="bti" cx="100" cy="100" r="1.8" fill="#ffffff"/>

      {/* Wordmark: ultra bold, heavy weight like "Welcome Back" in screenshot */}
      <text x="200" y="128"
            fontFamily="'Arial Black', 'Helvetica Neue', Arial, sans-serif"
            fontSize="82" fontWeight="700"
            letterSpacing="-3">
        <tspan fill="#ffffff">Derm</tspan><tspan fill="#4fa8f0">Aura</tspan>
      </text>

    </svg>
  );
}
