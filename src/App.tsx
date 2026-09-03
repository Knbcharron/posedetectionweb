import { useEffect, useRef, useState, useCallback } from "react";
import { DrawingUtils, PoseLandmarker } from "@mediapipe/tasks-vision";
import type { PoseLandmarkerResult } from "@mediapipe/tasks-vision";
import { usePoseLandmarker } from "./hooks/usePoseLandmarker";

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number>(0);
  const drawingUtilsRef = useRef<DrawingUtils | null>(null);
  const [hasCameraPermission, setHasCameraPermission] = useState<boolean | null>(null);

  // Async callback from the WebWorker
  const handlePoseResults = useCallback((results: PoseLandmarkerResult | null) => {
    if (!canvasRef.current || !videoRef.current) return;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext("2d");
    
    // Ensure canvas dimensions exactly match video dimensions
    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!drawingUtilsRef.current) {
        drawingUtilsRef.current = new DrawingUtils(ctx);
      }
      
      const drawingUtils = drawingUtilsRef.current;

      // Draw skeleton if landmarks are detected
      if (results && results.landmarks) {
        for (const landmarks of results.landmarks) {
          drawingUtils.drawLandmarks(landmarks, {
            radius: (data) => DrawingUtils.lerp(data.from!.z, -0.15, 0.1, 5, 1),
            color: "#FF0000",
          });
          drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
            color: "#00FF00",
            lineWidth: 3,
          });
        }
      }
    }
  }, []);

  const { detectPose, isReady } = usePoseLandmarker({ onResults: handlePoseResults });

  // Setup camera stream
  useEffect(() => {
    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            width: 640, 
            height: 480,
            facingMode: "environment"
          },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setHasCameraPermission(true);
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
        setHasCameraPermission(false);
      }
    };

    setupCamera();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  // Main capture loop - offloads processing to the WebWorker
  useEffect(() => {
    let lastVideoTime = -1;

    const captureLoop = () => {
      if (isReady && videoRef.current && videoRef.current.readyState >= 2) {
        const video = videoRef.current;
        // Only process when a new frame is available
        if (video.currentTime !== lastVideoTime) {
          lastVideoTime = video.currentTime;
          // Send to worker. The drawing happens in the handlePoseResults callback
          detectPose(video, performance.now());
        }
      }
      requestRef.current = requestAnimationFrame(captureLoop);
    };

    if (isReady) {
      requestRef.current = requestAnimationFrame(captureLoop);
    }

    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [isReady, detectPose]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', backgroundColor: '#000' }}>
      {!isReady && <p style={{ color: 'white', zIndex: 10 }}>Loading Background Worker...</p>}
      
      {hasCameraPermission === false && (
        <p style={{ color: 'red', zIndex: 10 }}>Camera access denied. Please grant permissions to use this app.</p>
      )}

      <div style={{ position: 'absolute', top: 0, left: 0, width: '100vw', height: '100vh', overflow: 'hidden', backgroundColor: 'black' }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            objectFit: 'cover'
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            pointerEvents: 'none'
          }}
        />
      </div>
    </div>
  );
}

export default App;
