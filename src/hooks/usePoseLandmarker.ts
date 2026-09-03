import { useEffect, useRef, useState, useCallback } from "react";
import type { PoseLandmarkerResult } from "@mediapipe/tasks-vision";

export interface PoseLandmarkerOptions {
  numPoses?: number;
  minPoseDetectionConfidence?: number;
  minPosePresenceConfidence?: number;
  minTrackingConfidence?: number;
}

interface UsePoseLandmarkerProps {
  options?: PoseLandmarkerOptions;
  onResults?: (results: PoseLandmarkerResult | null, timestamp: number) => void;
}

export const usePoseLandmarker = ({ options, onResults }: UsePoseLandmarkerProps = {}) => {
  const [isReady, setIsReady] = useState(false);
  const workerRef = useRef<Worker | null>(null);
  
  // Keep latest callback in a ref to prevent unnecessary effect re-runs
  const onResultsRef = useRef(onResults);
  useEffect(() => {
    onResultsRef.current = onResults;
  }, [onResults]);

  useEffect(() => {
    const worker = new Worker(new URL('../workers/poseWorker.ts', import.meta.url), {
      type: 'module'
    });

    worker.onmessage = (e) => {
      const { type, results, timestamp } = e.data;
      if (type === "READY") {
        setIsReady(true);
      } else if (type === "RESULTS") {
        if (onResultsRef.current) {
          onResultsRef.current(results, timestamp);
        }
      }
    };

    // Send initialization options
    worker.postMessage({ type: "INIT", options: options || {} });

    workerRef.current = worker;

    return () => {
      worker.postMessage({ type: "CLOSE" });
      worker.terminate();
    };
  }, []); // Run once. Dynamic option updates should be handled via setOptions if needed.

  const setOptions = useCallback((newOptions: PoseLandmarkerOptions) => {
    if (workerRef.current && isReady) {
      workerRef.current.postMessage({ type: "SET_OPTIONS", options: newOptions });
    }
  }, [isReady]);

  const detectPose = useCallback(async (videoElement: HTMLVideoElement, timestamp: number) => {
    if (!workerRef.current || !isReady) return;
    try {
      const imageBitmap = await createImageBitmap(videoElement);
      workerRef.current.postMessage(
        { type: "DETECT", image: imageBitmap, timestamp }, 
        [imageBitmap] // Transfer ownership to worker for zero-copy
      );
    } catch (error) {
      console.error("Error creating ImageBitmap for worker:", error);
    }
  }, [isReady]);

  return { detectPose, isReady, setOptions };
};
