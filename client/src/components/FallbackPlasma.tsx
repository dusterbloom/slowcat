"use client";

import React from "react";

interface FallbackPlasmaProps {
  isDarkMode: boolean;
}

/**
 * FallbackPlasma
 * A lightweight, pure CSS "plasma-like" background using animated gradients.
 * This is used when WebGL is disabled or performance degradation is detected.
 */
export function FallbackPlasma({ isDarkMode }: FallbackPlasmaProps) {
  return (
    <div
      className={`absolute inset-0 overflow-hidden ${
        isDarkMode ? "bg-black" : "bg-white"
      }`}
    >
      <div
        className={`w-full h-full animate-gradientMove ${
          isDarkMode
            ? "bg-gradient-to-br from-purple-900 via-orange-900 to-red-900"
            : "bg-gradient-to-br from-pink-200 via-orange-200 to-yellow-200"
        } bg-[length:400%_400%] opacity-70`}
        style={{
          animation: "gradientMove 12s ease infinite",
        }}
      />
      <style jsx>{`
        @keyframes gradientMove {
          0% {
            background-position: 0% 50%;
          }
          50% {
            background-position: 100% 50%;
          }
          100% {
            background-position: 0% 50%;
          }
        }
      `}</style>
    </div>
  );
}

export default FallbackPlasma;