import { useEffect, useRef, useState } from "react";
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import type { PoseLandmarkerResult } from "@mediapipe/tasks-vision";

export const usePoseLandmarker = () => {
  const [isReady, setIsReady] = useState(false);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);

  useEffect(() => {
    let active = true;

    const initializePoseLandmarker = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        if (!active) return;

        const landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/pose_landmarker_full.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.7,
          minPosePresenceConfidence: 0.7,
          minTrackingConfidence: 0.7,
        });

        if (!active) {
          landmarker.close();
          return;
        }

        poseLandmarkerRef.current = landmarker;
        setIsReady(true);
      } catch (error) {
        console.error("Error initializing Pose Landmarker:", error);
      }
    };

    initializePoseLandmarker();

    return () => {
      active = false;
      if (poseLandmarkerRef.current) {
        poseLandmarkerRef.current.close();
      }
    };
  }, []);

  const detectPose = (
    videoElement: HTMLVideoElement,
    timestamp: number
  ): PoseLandmarkerResult | null => {
    if (!poseLandmarkerRef.current || !isReady) {
      return null;
    }
    
    try {
      return poseLandmarkerRef.current.detectForVideo(videoElement, timestamp);
    } catch (error) {
      console.error("Error detecting pose:", error);
      return null;
    }
  };

  return { detectPose, isReady };
};
