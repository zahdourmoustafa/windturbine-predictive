"use client";

import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface PredictionConfidenceGaugeProps {
  confidence: number; // Value between 0 and 1
  title?: string;
  description?: string;
}

export function PredictionConfidenceGauge({
  confidence,
  title = "Overall Prediction Confidence",
  description = "Gauge showing the overall model prediction confidence",
}: PredictionConfidenceGaugeProps) {
  // Ensure confidence is within 0-1 range
  const normalizedConfidence = Math.max(0, Math.min(1, confidence));
  const percentage = Math.round(normalizedConfidence * 100);
  
  // Determine color based on confidence level
  const getColor = () => {
    if (percentage >= 70) return "text-green-500";
    if (percentage >= 40) return "text-yellow-500";
    return "text-red-500";
  };

  // Calculate gauge rotation angle
  const gaugeAngle = 180 * normalizedConfidence;

  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center justify-center">
          {/* Gauge display */}
          <div className="relative w-44 h-24 mb-2">
            {/* Gauge background */}
            <div className="absolute w-44 h-44 bottom-0 rounded-t-full overflow-hidden border-t border-l border-r border-gray-300 dark:border-gray-700">
              <div className="absolute w-44 h-44 bottom-0 bg-gray-100 dark:bg-gray-800 rounded-t-full"></div>
            </div>
            
            {/* Gauge indicator */}
            <div 
              className="absolute w-1 h-24 bottom-0 left-[88px] origin-bottom bg-black dark:bg-white"
              style={{ 
                transform: `rotate(${gaugeAngle - 90}deg)`,
                transformOrigin: "bottom center" 
              }}
            />
            
            {/* Gauge markers */}
            <div className="absolute w-44 h-44 bottom-0">
              {[0, 30, 60, 90, 120, 150, 180].map((deg) => (
                <div 
                  key={deg}
                  className="absolute w-1 h-3 bg-gray-400 dark:bg-gray-600"
                  style={{ 
                    bottom: "0px",
                    left: "50%",
                    transform: `translateX(-50%) rotate(${deg - 90}deg)`,
                    transformOrigin: "bottom center",
                  }}
                />
              ))}
            </div>
          </div>
          
          {/* Percentage display */}
          <div className={`text-3xl font-bold mt-2 ${getColor()}`}>
            {percentage}%
          </div>
          
          {/* Scale labels */}
          <div className="w-44 flex justify-between text-xs text-gray-500 mt-1">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 