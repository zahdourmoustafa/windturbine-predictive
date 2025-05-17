"use client";

import { useState, ChangeEvent, FormEvent } from "react";
import { AlertCircle, CheckCircle2, Loader2, UploadCloud } from "lucide-react";

// Assuming shadcn/ui components are available. If not, you'll need to add them:
// npx shadcn-ui@latest add card button input table alert progress
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input"; // Or a custom file input wrapper
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
// import { Progress } from "@/components/ui/progress"; // For progress bar if needed

// Define interfaces for the API response structure (matching Pydantic models)
interface SensorDiagnosis {
  sensor: string;
  predicted_state: string;
  failure_mode: string;
  associated_component: string;
}

interface Metrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  confusion_matrix?: number[][];
}

interface LossMetrics {
  total_loss?: number;
  detection_loss?: number;
  anomaly_loss?: number;
}

interface PredictionResult {
  filename: string;
  error?: string;
  file_level_prediction?: string;
  confidence_for_decision?: number;
  avg_overall_fault_probability_for_file?: number;
  was_healthy_override_applied?: boolean;
  per_sensor_diagnoses?: SensorDiagnosis[];
  overall_detection_metrics?: Metrics;
  average_per_sensor_f1?: number;
  per_sensor_f1_scores?: { [key: string]: number };
  loss_metrics_vs_gt?: LossMetrics;
  raw_data_preview_segment?: number[];
  avg_sensor_probs_across_all_windows?: { [key: string]: number };
  mc_dropout_used?: boolean;
  avg_overall_fault_uncertainty?: number;
  avg_sensor_anomaly_uncertainty_per_sensor?: { [key: string]: number };
}

interface ApiResponse {
  data?: PredictionResult;
  message?: string;
  success: boolean;
}

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileNameDisplay, setFileNameDisplay] =
    useState<string>("No file selected");

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      if (selectedFile.name.endsWith(".mat")) {
        setFile(selectedFile);
        setFileNameDisplay(selectedFile.name);
        setError(null); // Clear previous error
      } else {
        setFile(null);
        setFileNameDisplay("Invalid file type. Please upload a .mat file.");
        setError("Invalid file type. Please upload a .mat file.");
      }
    }
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file) {
      setError("Please select a .mat file to upload.");
      return;
    }

    setIsLoading(true);
    setPrediction(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/api/v1/predict", {
        method: "POST",
        body: formData,
      });

      const result = (await response.json()) as ApiResponse;

      if (!response.ok || !result.success) {
        const errorMessage =
          result.message ||
          result.data?.error ||
          `HTTP error! status: ${response.status}`;
        throw new Error(errorMessage);
      }

      if (result.data) {
        setPrediction(result.data);
      } else {
        throw new Error(
          result.message || "Prediction data not found in response."
        );
      }
    } catch (err: any) {
      console.error("Submission error:", err);
      setError(err.message || "An unknown error occurred during prediction.");
      setPrediction(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4 md:p-8">
      <Card className="max-w-3xl mx-auto">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-center">
            Gearbox Fault Diagnosis
          </CardTitle>
          <CardDescription className="text-center">
            Upload a .mat sensor data file to predict potential faults and
            analyze sensor status.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <label
                htmlFor="file-upload"
                className="block text-sm font-medium text-gray-700"
              >
                Select .mat file
              </label>
              <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                <div className="space-y-1 text-center">
                  <UploadCloud className="mx-auto h-12 w-12 text-gray-400" />
                  <div className="flex text-sm text-gray-600">
                    <label
                      htmlFor="file-upload"
                      className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500"
                    >
                      <span>Upload a file</span>
                      <Input
                        id="file-upload"
                        name="file-upload"
                        type="file"
                        className="sr-only"
                        onChange={handleFileChange}
                        accept=".mat"
                      />
                    </label>
                    <p className="pl-1">or drag and drop</p>
                  </div>
                  <p className="text-xs text-gray-500">MAT files up to 50MB</p>
                  <p
                    className="text-sm text-gray-700 pt-2"
                    id="filename-display"
                  >
                    {fileNameDisplay}
                  </p>
                </div>
              </div>
            </div>
            <Button
              type="submit"
              className="w-full"
              disabled={isLoading || !file}
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Analyzing...
                </>
              ) : (
                "Analyze Gearbox Data"
              )}
            </Button>
          </form>
        </CardContent>
        {error && (
          <CardFooter>
            <Alert variant="destructive" className="w-full">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          </CardFooter>
        )}
      </Card>

      {isLoading && (
        <div className="mt-8 text-center">
          <Loader2 className="mx-auto h-12 w-12 text-indigo-600 animate-spin" />
          <p className="mt-2 text-lg font-medium">
            Processing your file, please wait...
          </p>
          {/* <Progress value={33} className="w-1/2 mx-auto mt-2" /> // Example if you had progress */}
        </div>
      )}

      {prediction && !isLoading && (
        <Card className="mt-8">
          <CardHeader>
            <CardTitle className="flex items-center">
              <CheckCircle2 className="h-6 w-6 text-green-500 mr-2" />
              Analysis Results for: {prediction.filename}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Overall Prediction Section */}
            <section>
              <h3 className="text-lg font-semibold mb-2">
                Overall File Prediction
              </h3>
              <Alert
                variant={
                  prediction.file_level_prediction === "FAULTY"
                    ? "destructive"
                    : "default"
                }
                className={
                  prediction.file_level_prediction === "HEALTHY"
                    ? "bg-green-50 border-green-300 text-green-700"
                    : ""
                }
              >
                <AlertTitle className="font-bold text-xl">
                  {prediction.file_level_prediction || "N/A"}
                </AlertTitle>
                <AlertDescription>
                  {prediction.file_level_prediction === "FAULTY"
                    ? `Average fault probability: ${(
                        (prediction.avg_overall_fault_probability_for_file ||
                          0) * 100
                      ).toFixed(1)}%`
                    : `Confidence for Healthy State: ${(
                        (1.0 -
                          (prediction.avg_overall_fault_probability_for_file ||
                            0)) *
                        100
                      ).toFixed(1)}%`}
                  {prediction.was_healthy_override_applied && (
                    <span className="block text-sm font-medium">
                      {" "}
                      (Note: Overall status was overridden to HEALTHY based on
                      sensor health analysis)
                    </span>
                  )}
                </AlertDescription>
              </Alert>
            </section>

            {/* Per-Sensor Diagnosis Table Section */}
            {prediction.per_sensor_diagnoses &&
              prediction.per_sensor_diagnoses.length > 0 && (
                <section>
                  <h3 className="text-lg font-semibold mb-2">
                    Per-Sensor Detailed Diagnosis
                  </h3>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Sensor</TableHead>
                        <TableHead>Predicted State</TableHead>
                        <TableHead>Failure Mode</TableHead>
                        <TableHead>Associated Component</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {prediction.per_sensor_diagnoses.map((diag, index) => (
                        <TableRow
                          key={`${diag.sensor}-${diag.failure_mode}-${index}`}
                        >
                          <TableCell
                            className={diag.sensor ? "font-medium" : ""}
                          >
                            {diag.sensor}
                          </TableCell>
                          <TableCell
                            className={
                              diag.predicted_state
                                ? diag.predicted_state === "Damaged"
                                  ? "text-red-600 font-semibold"
                                  : "text-green-600 font-semibold"
                                : ""
                            }
                          >
                            {diag.predicted_state}
                          </TableCell>
                          <TableCell>{diag.failure_mode}</TableCell>
                          <TableCell>{diag.associated_component}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </section>
              )}

            {/* Metrics Section - Display only if available */}
            {(prediction.overall_detection_metrics ||
              prediction.loss_metrics_vs_gt) && (
              <section>
                <h3 className="text-lg font-semibold mb-2">
                  Performance Metrics (if Ground Truth was available for this
                  file)
                </h3>
                {prediction.overall_detection_metrics && (
                  <div className="mb-4 p-4 border rounded-md">
                    <h4 className="text-md font-semibold mb-1">
                      Overall Detection Metrics:
                    </h4>
                    <p>
                      Accuracy:{" "}
                      {prediction.overall_detection_metrics.accuracy?.toFixed(
                        4
                      ) || "N/A"}
                    </p>
                    <p>
                      Precision:{" "}
                      {prediction.overall_detection_metrics.precision?.toFixed(
                        4
                      ) || "N/A"}
                    </p>
                    <p>
                      Recall:{" "}
                      {prediction.overall_detection_metrics.recall?.toFixed(
                        4
                      ) || "N/A"}
                    </p>
                    <p>
                      F1 Score:{" "}
                      {prediction.overall_detection_metrics.f1?.toFixed(4) ||
                        "N/A"}
                    </p>
                    {prediction.overall_detection_metrics.confusion_matrix && (
                      <div className="mt-2">
                        <p className="font-medium">
                          Confusion Matrix (GT vs Pred):
                        </p>
                        <pre className="bg-gray-100 p-2 rounded-md text-sm">
                          {`  Predicted:   Healthy  Faulty
Actual Healthy: [[${prediction.overall_detection_metrics.confusion_matrix[0][0]},      ${prediction.overall_detection_metrics.confusion_matrix[0][1]}],
Actual Faulty:   [${prediction.overall_detection_metrics.confusion_matrix[1][0]},      ${prediction.overall_detection_metrics.confusion_matrix[1][1]}]]`}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
                {prediction.per_sensor_f1_scores &&
                  Object.keys(prediction.per_sensor_f1_scores).length > 0 && (
                    <div className="mb-4 p-4 border rounded-md">
                      <h4 className="text-md font-semibold mb-1">
                        Per-Sensor F1 Scores:
                      </h4>
                      {Object.entries(prediction.per_sensor_f1_scores).map(
                        ([sensor, f1]) => (
                          <p key={sensor}>
                            {sensor}: {f1.toFixed(4)}
                          </p>
                        )
                      )}
                      {prediction.average_per_sensor_f1 && (
                        <p className="font-semibold mt-1">
                          Average Per-Sensor F1:{" "}
                          {prediction.average_per_sensor_f1.toFixed(4)}
                        </p>
                      )}
                    </div>
                  )}
                {prediction.loss_metrics_vs_gt && (
                  <div className="p-4 border rounded-md">
                    <h4 className="text-md font-semibold mb-1">
                      Loss Metrics (vs GT):
                    </h4>
                    <p>
                      Total Loss:{" "}
                      {prediction.loss_metrics_vs_gt.total_loss?.toFixed(4) ||
                        "N/A"}
                    </p>
                    <p>
                      Detection Loss:{" "}
                      {prediction.loss_metrics_vs_gt.detection_loss?.toFixed(
                        4
                      ) || "N/A"}
                    </p>
                    <p>
                      Anomaly Loss:{" "}
                      {prediction.loss_metrics_vs_gt.anomaly_loss?.toFixed(4) ||
                        "N/A"}
                    </p>
                  </div>
                )}
                {!prediction.overall_detection_metrics &&
                  !prediction.loss_metrics_vs_gt && (
                    <p className="text-sm text-gray-600">
                      No ground truth data was found for this specific file to
                      calculate detailed performance metrics.
                    </p>
                  )}
              </section>
            )}

            {/* Other info like MC dropout if used */}
            {prediction.mc_dropout_used && (
              <section>
                <h3 className="text-lg font-semibold mb-2">
                  Model Uncertainty (MC Dropout)
                </h3>
                <div className="p-4 border rounded-md">
                  <p>
                    Average Overall Fault Uncertainty:{" "}
                    {prediction.avg_overall_fault_uncertainty?.toFixed(4) ||
                      "N/A"}
                  </p>
                  {/* Could also list per-sensor uncertainty if desired */}
                </div>
              </section>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
