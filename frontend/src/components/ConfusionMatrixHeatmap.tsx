import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { PlotParams } from 'react-plotly.js';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface ConfusionMatrixHeatmapProps {
  confusionMatrix: number[][];
}

export function ConfusionMatrixHeatmap({ confusionMatrix }: ConfusionMatrixHeatmapProps) {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Don't render on server-side
  if (!isMounted || !confusionMatrix) {
    return <div className="h-40 w-full bg-gray-100 animate-pulse rounded-md"></div>;
  }

  // Extract values for better readability
  const tn = confusionMatrix[0][0]; // True Negative
  const fp = confusionMatrix[0][1]; // False Positive
  const fn = confusionMatrix[1][0]; // False Negative
  const tp = confusionMatrix[1][1]; // True Positive

  // Calculate accuracy for annotation text
  const total = tn + fp + fn + tp;
  const accuracy = total > 0 ? ((tn + tp) / total).toFixed(2) : 'N/A';

  // Prepare data for heatmap
  const data: Partial<Plotly.PlotData>[] = [
    {
      z: confusionMatrix,
      x: ['Healthy', 'Faulty'],
      y: ['Healthy', 'Faulty'],
      type: 'heatmap',
      colorscale: [
        [0, 'rgb(247, 251, 255)'],
        [0.5, 'rgb(107, 174, 214)'],
        [1, 'rgb(8, 81, 156)']
      ],
      showscale: false,
      hoverinfo: 'text',
      text: [
        [`True Negative: ${tn}`, `False Positive: ${fp}`],
        [`False Negative: ${fn}`, `True Positive: ${tp}`]
      ]
    }
  ];

  // Layout configuration
  const layout: Partial<Plotly.Layout> = {
    title: {
      text: `Confusion Matrix (Accuracy: ${accuracy})`,
      font: { size: 16 }
    },
    annotations: [
      {
        x: 'Healthy',
        y: 'Healthy',
        text: tn.toString(),
        font: { color: 'black' },
        showarrow: false
      },
      {
        x: 'Faulty',
        y: 'Healthy',
        text: fp.toString(),
        font: { color: 'black' },
        showarrow: false
      },
      {
        x: 'Healthy',
        y: 'Faulty',
        text: fn.toString(),
        font: { color: 'black' },
        showarrow: false
      },
      {
        x: 'Faulty',
        y: 'Faulty',
        text: tp.toString(),
        font: { color: 'black' },
        showarrow: false
      }
    ],
    xaxis: {
      title: 'Predicted',
      tickangle: 0
    },
    yaxis: {
      title: 'Actual',
      automargin: true
    },
    margin: { l: 60, r: 25, b: 60, t: 40 },
    width: 400,
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
      />
    </div>
  );
} 