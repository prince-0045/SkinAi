import React, { useState, useRef, useEffect } from 'react';
import { MoveHorizontal } from 'lucide-react';

export default function ImageComparison({ beforeImage, afterImage }) {
    const [sliderPosition, setSliderPosition] = useState(50);
    const containerRef = useRef(null);

    const handleMouseMove = (e) => {
        if (!containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
        const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
        setSliderPosition((x / rect.width) * 100);
    };

    const handleTouchMove = (e) => {
        if (!containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
        const x = Math.max(0, Math.min(e.touches[0].clientX - rect.left, rect.width));
        setSliderPosition((x / rect.width) * 100);
    };

    return (
        <div
            ref={containerRef}
            className="relative w-full h-80 rounded-2xl overflow-hidden cursor-col-resize group select-none"
            onMouseMove={handleMouseMove}
            onTouchMove={handleTouchMove}
        >
            {/* After Image (Background) */}
            <img
                src={afterImage}
                alt="After"
                className="absolute top-0 left-0 w-full h-full object-cover"
            />
            <div className="absolute top-4 right-4 bg-black/50 text-white px-2 py-1 rounded text-xs font-bold">AFTER</div>

            {/* Before Image (Clipped) */}
            <div
                className="absolute top-0 left-0 h-full overflow-hidden border-r-4 border-white shadow-xl"
                style={{ width: `${sliderPosition}%` }}
            >
                <img
                    src={beforeImage}
                    alt="Before"
                    className="absolute top-0 left-0 max-w-none h-full object-cover"
                    style={{ width: containerRef.current ? containerRef.current.offsetWidth : '100%' }}
                />
                <div className="absolute top-4 left-4 bg-black/50 text-white px-2 py-1 rounded text-xs font-bold">BEFORE</div>
            </div>

            {/* Slider Handle */}
            <div
                className="absolute top-0 bottom-0 w-1 bg-transparent cursor-col-resize flex items-center justify-center pointer-events-none"
                style={{ left: `${sliderPosition}%` }}
            >
                <div className="w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center transform -translate-x-1/2">
                    <MoveHorizontal className="w-4 h-4 text-medical-600" />
                </div>
            </div>
        </div>
    );
}
