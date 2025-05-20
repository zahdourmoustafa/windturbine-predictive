import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SensorF1ScoreBarChartProps {
  sensorScores: Record<string, number>; // Format: { "AN3": 0.9426, "AN4": 0.9406, ... }
}

export function SensorF1ScoreBarChart({ sensorScores }: SensorF1ScoreBarChartProps) {
  const [isMounted, setIsMounted] = useState(false);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Don't render on server-side
  if (!isMounted || !sensorScores) {
    return <div className="h-40 w-full bg-gray-100 animate-pulse rounded-md"></div>;
  }

  // If we want to use a fallback instead of Plotly, we can switch this to true for testing
  if (hasError || Object.keys(sensorScores).length === 0) {
    return <FallbackBarChart sensorScores={sensorScores} />;
  }

  try {
    // Convert data to arrays for Plotly
    const sensorIds = Object.keys(sensorScores);
    const f1Scores = Object.values(sensorScores);

    // Sort data by sensor ID (assuming they're in format AN3, AN4, etc.)
    const sortedIndices = sensorIds.map((_, index) => index)
      .sort((a, b) => {
        const idA = sensorIds[a];
        const idB = sensorIds[b];
        return idA.localeCompare(idB, undefined, { numeric: true });
      });

    const sortedSensorIds = sortedIndices.map(i => sensorIds[i]);
    const sortedF1Scores = sortedIndices.map(i => f1Scores[i]);

    // Prepare data for horizontal bar chart
    const data: Partial<Plotly.PlotData>[] = [
      {
        y: sortedSensorIds,
        x: sortedF1Scores,
        type: 'bar',
        orientation: 'h',
        marker: {
          color: 'rgb(8, 81, 156)',
          opacity: 0.8
        },
        text: sortedF1Scores.map(score => score.toFixed(4)),
        textposition: 'auto',
        hoverinfo: 'text',
        hovertext: sortedF1Scores.map((score, i) => `${sortedSensorIds[i]}: ${score.toFixed(4)}`)
      }
    ];

    // Layout configuration
    const layout: Partial<Plotly.Layout> = {
      title: {
        text: 'Per-Sensor F1 Score',
        font: { size: 16 }
      },
      xaxis: {
        title: 'F1-Score',
        range: [0, 1],
        tickformat: '.2f'
      },
      yaxis: {
        title: 'Sensor ID',
        automargin: true
      },
      margin: { l: 90, r: 25, b: 60, t: 40 },
      height: 350,
      autosize: true
    };

    const config: Partial<Plotly.Config> = {
      displayModeBar: false,
      responsive: true
    };

    return (
      <div className="w-full">
        <Plot
          data={data}
          layout={layout}
          config={config}
          className="w-full"
          onError={() => {
            setHasError(true);
          }}
        />
      </div>
    );
  } catch (error) {
    // If Plotly fails to render, show the fallback
    return <FallbackBarChart sensorScores={sensorScores} />;
  }
}

// Simple fallback component that doesn't require Plotly
function FallbackBarChart({ sensorScores }: SensorF1ScoreBarChartProps) {
  // Sort sensors by ID
  const sortedScores = Object.entries(sensorScores)
    .sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true }));

  return (
    <div className="w-full border rounded-md p-4">
      <h3 className="text-base font-semibold mb-3">Per-Sensor F1 Score</h3>
      <div className="space-y-2">
        {sortedScores.map(([sensorId, score]) => (
          <div key={sensorId} className="flex items-center gap-2">
            <span className="w-16 font-medium text-sm">{sensorId}:</span>
            <div className="flex-1 bg-gray-100 rounded-full h-5">
              <div 
                className="bg-blue-800 rounded-full h-5"
                style={{ width: `${score * 100}%` }}
              ></div>
            </div>
            <span className="text-sm w-16 text-right">{score.toFixed(4)}</span>
          </div>
        ))}
      </div>
    </div>
  );
} 