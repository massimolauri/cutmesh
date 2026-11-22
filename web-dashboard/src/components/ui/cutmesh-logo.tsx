"use client";

import React from "react";

interface CutmeshLogoProps {
  size?: number;
  className?: string;
}

/**
 * CUTMESH Swiss Editorial SVG Emblem — Frameless Interlocking Blades (Warm Coral):
 * - Upper-Left Blade (Node A): Warm Cream (#FFF7ED / currentColor)
 * - Diagonal Precision Cut Plane: Warm Coral (#FB7185)
 * - Lower-Right Blade (Node B): Warm Apricot (#FDBA74)
 * - Zero frame, zero card, zero gradients, zero glow.
 */
export function CutmeshLogo({ size = 32, className = "" }: CutmeshLogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="5 7.5 24.5 22"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="CUTMESH"
    >
      <polygon
        points="6.2,15.2 13.2,8.2 16.7,11.7 14.2,14.2 13.2,13.2 11.2,15.2 12.7,16.7 20.4,9 25.4,9 12.7,21.7"
        fill="currentColor"
      />
      <polygon
        points="28.2,10 10.7,27.5 15.7,27.5 23.7,19.5 23.7,22 22.2,23.5 21.2,22.5 18.7,25 22.2,28.5 28.2,22.5"
        fill="#FDBA74"
      />
      <polygon points="8.4,27.0 27.0,8.4 27.9,9.3 9.3,27.9" fill="#FB7185" />
    </svg>
  );
}

interface CutmeshWordmarkProps {
  className?: string;
  showIcon?: boolean;
  iconSize?: number;
}

/**
 * CUTMESH Brand Wordmark where the Warm Coral Interlocking Blades emblem
 * replaces the letter "S" inside CUTME[S]H (frameless, razor-sharp inline glyph).
 */
export function CutmeshWordmark({
  className = "text-sm",
  showIcon = false,
  iconSize = 22
}: CutmeshWordmarkProps) {
  return (
    <span
      className={`inline-flex items-center gap-2 font-bold tracking-[0.14em] uppercase text-foreground select-none leading-none ${className}`}
      aria-label="CUTMESH"
    >
      {showIcon && <CutmeshLogo size={iconSize} className="shrink-0" />}
      <span className="inline-flex items-center">
        <span>CUTME</span>
        <svg
          viewBox="5 7.5 24.5 22"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="inline-block h-[1.08em] w-auto mr-[0.10em] -ml-[0.04em] translate-y-[0.02em]"
          aria-hidden="true"
        >
          <polygon
            points="6.2,15.2 13.2,8.2 16.7,11.7 14.2,14.2 13.2,13.2 11.2,15.2 12.7,16.7 20.4,9 25.4,9 12.7,21.7"
            fill="currentColor"
          />
          <polygon
            points="28.2,10 10.7,27.5 15.7,27.5 23.7,19.5 23.7,22 22.2,23.5 21.2,22.5 18.7,25 22.2,28.5 28.2,22.5"
            fill="#FDBA74"
          />
          <polygon points="8.4,27.0 27.0,8.4 27.9,9.3 9.3,27.9" fill="#FB7185" />
        </svg>
        <span>H</span>
      </span>
    </span>
  );
}

/**
 * Subtle background watermark of the Warm Coral Cutmesh Interlocking Blades emblem
 * for the login screen (no neon, no gradients).
 */
export function CutmeshWatermark({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="5 7.5 24.5 22"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      <polygon
        points="6.2,15.2 13.2,8.2 16.7,11.7 14.2,14.2 13.2,13.2 11.2,15.2 12.7,16.7 20.4,9 25.4,9 12.7,21.7"
        fill="currentColor"
      />
      <polygon
        points="28.2,10 10.7,27.5 15.7,27.5 23.7,19.5 23.7,22 22.2,23.5 21.2,22.5 18.7,25 22.2,28.5 28.2,22.5"
        fill="#FDBA74"
      />
      <polygon points="8.4,27.0 27.0,8.4 27.9,9.3 9.3,27.9" fill="#FB7185" />
    </svg>
  );
}

/**
 * Animated Frameless Cutmesh Loader:
 * - The central Warm Coral (#FB7185) 45-degree laser line sweeps across the split seam
 *   while the two interlocking blades subtly separate and lock back into place.
 */
export function CutmeshLoader({
  label,
  fullScreen = true
}: {
  label?: string;
  fullScreen?: boolean;
}) {
  return (
    <div
      className={
        fullScreen
          ? "min-h-screen w-full bg-background text-foreground flex flex-col items-center justify-center gap-5 select-none"
          : "w-full py-20 flex flex-col items-center justify-center gap-4 select-none"
      }
      role="status"
      aria-label={label || "Loading Cutmesh"}
    >
      <svg
        width={68}
        height={68}
        viewBox="4 6.5 26.5 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <polygon
          className="animate-cutmesh-upper"
          points="6.2,15.2 13.2,8.2 16.7,11.7 14.2,14.2 13.2,13.2 11.2,15.2 12.7,16.7 20.4,9 25.4,9 12.7,21.7"
          fill="currentColor"
        />
        <polygon
          className="animate-cutmesh-lower"
          points="28.2,10 10.7,27.5 15.7,27.5 23.7,19.5 23.7,22 22.2,23.5 21.2,22.5 18.7,25 22.2,28.5 28.2,22.5"
          fill="#FDBA74"
        />
        <line
          x1="8.8"
          y1="27.4"
          x2="27.4"
          y2="8.8"
          stroke="#FB7185"
          strokeOpacity="0.22"
          strokeWidth="1.35"
        />
        <line
          className="animate-cutmesh-laser"
          x1="8.8"
          y1="27.4"
          x2="27.4"
          y2="8.8"
          stroke="#FB7185"
          strokeWidth="1.75"
          strokeLinecap="square"
        />
      </svg>

      <div className="flex flex-col items-center gap-1.5">
        <CutmeshWordmark className="text-sm tracking-[0.22em]" />
        {label && (
          <span className="text-[11px] font-mono text-muted-foreground tracking-wider uppercase">
            {label}
          </span>
        )}
      </div>
    </div>
  );
}
