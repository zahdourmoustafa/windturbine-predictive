"use client";

import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface PredictionConfidencePieProps {
  confidence: number; // Value between 0 and 1
  title?: string;
  description?: string;
}

export function PredictionConfidencePie({
  confidence,
  title = "Overall Prediction Confidence",
  description = "Pie chart showing the overall model prediction confidence",
}: PredictionConfidencePieProps) {
  // Ensure confidence is within 0-1 range
  const normalizedConfidence = Math.max(0, Math.min(1, confidence));
  const percentage = Math.round(normalizedConfidence * 100);

  // Determine color based on confidence level
  const getColor = () => {
    if (percentage >= 70) return "#22c55e"; // green-500
    if (percentage >= 40) return "#eab308"; // yellow-500
    return "#ef4444"; // red-500
  };

  const color = getColor();

  // Calculate the circle's circumference and stroke-dasharray
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const strokeDasharray = `${circumference * normalizedConfidence} ${
    circumference * (1 - normalizedConfidence)
  }`;

  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center justify-center">
          {/* Pie/Circle visualization */}
          <div className="relative w-32 h-32">
            {/* Background circle */}
            <svg className="w-full h-full" viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r={radius}
                fill="none"
                stroke="#e5e7eb"
                strokeWidth="10"
                className="dark:stroke-gray-700"
              />
              {/* Foreground circle (the progress) */}
              <circle
                cx="50"
                cy="50"
                r={radius}
                fill="none"
                stroke={color}
                strokeWidth="10"
                strokeDasharray={strokeDasharray}
                strokeLinecap="round"
                transform="rotate(-90 50 50)"
              />
            </svg>

            {/* Percentage display in center */}
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-2xl font-bold" style={{ color }}>
                {percentage}%
              </span>
            </div>
          </div>

          {/* Legend */}
          <div className="mt-4 flex gap-3 text-sm">
            <div className="flex items-center">
              <div
                className="w-3 h-3 rounded-full mr-1"
                style={{ backgroundColor: color }}
              ></div>
              <span>Confident ({percentage}%)</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 rounded-full bg-gray-200 dark:bg-gray-700 mr-1"></div>
              <span>Uncertain ({100 - percentage}%)</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
