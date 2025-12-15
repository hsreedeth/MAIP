import React from 'react';

interface ProgressBarProps {
  completed: number;
  total: number;
  width?: number;
  label?: string;
  className?: string;
  variant?: 'normal' | 'slim';
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ 
  completed, 
  total, 
  width = 20, 
  label, 
  className = "",
  variant = 'normal'
}) => {
  const percentage = Math.round((completed / Math.max(total, 1)) * 100);
  const filledChars = Math.round((completed / Math.max(total, 1)) * width);
  const emptyChars = width - filledChars;
  
  const bar = `[${'#'.repeat(filledChars)}${'.'.repeat(emptyChars)}]`;
  
  const colorClass = percentage === 100 ? 'text-green-500' : 'text-yellow-500';

  if (variant === 'slim') {
    return (
      <span className={`font-mono whitespace-pre ${className}`}>
        <span className={colorClass}>{bar}</span> {percentage.toString().padStart(3)}%
      </span>
    );
  }

  return (
    <div className={`font-mono whitespace-pre ${className}`}>
      {label && <div className="mb-1 uppercase text-xs tracking-wider opacity-80">{label}</div>}
      <div className="flex items-center">
        <span className={`${colorClass} mr-2`}>{bar}</span>
        <span className="text-sm">{percentage}%</span>
      </div>
    </div>
  );
};