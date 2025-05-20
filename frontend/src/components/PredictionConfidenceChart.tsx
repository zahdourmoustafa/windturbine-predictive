"use client";

import React, { useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { PredictionConfidenceGauge } from "./PredictionConfidenceGauge";
import { PredictionConfidencePie } from "./PredictionConfidencePie";

interface PredictionConfidenceChartProps {
  confidence: number; // Value between 0 and 1
  title?: string;
  description?: string;
}

type ChartType = "gauge" | "pie";

export function PredictionConfidenceChart({
  confidence,
  title = "Overall Prediction Confidence",
  description = "Visualization of the overall model prediction confidence",
}: PredictionConfidenceChartProps) {
  const [chartType, setChartType] = useState<ChartType>("gauge");

  return (
    <Card className="w-full">
      <CardHeader className="pb-2 flex flex-row items-center justify-between">
        <CardTitle className="text-lg">{title}</CardTitle>
        <Select
          value={chartType}
          onValueChange={(value) => setChartType(value as ChartType)}
        >
          <SelectTrigger className="w-[130px]">
            <SelectValue placeholder="Chart Type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="gauge">Gauge</SelectItem>
            <SelectItem value="pie">Pie Chart</SelectItem>
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent>
        {chartType === "gauge" ? (
          <PredictionConfidenceGauge 
            confidence={confidence} 
            title="" 
            description="" 
          />
        ) : (
          <PredictionConfidencePie 
            confidence={confidence} 
            title="" 
            description="" 
          />
        )}
      </CardContent>
    </Card>
  );
} 