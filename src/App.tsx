import { useEffect, useRef, useState } from "react";
import { DrawingUtils, PoseLandmarker } from "@mediapipe/tasks-vision";
import { usePoseLandmarker } from "./hooks/usePoseLandmarker";

function App() {
  const { detectPose, isReady } = usePoseLandmarker();
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number>(0);
  const [hasCameraPermission, setHasCameraPermission] = useState<boolean | null>(null);

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

  // Main rendering loop
  useEffect(() => {
    let lastVideoTime = -1;
    let drawingUtils: DrawingUtils | null = null;

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) {
        drawingUtils = new DrawingUtils(ctx);
      }
    }

    const renderLoop = () => {
      if (
        isReady &&
        videoRef.current &&
        videoRef.current.readyState >= 2 &&
        canvasRef.current &&
        drawingUtils
      ) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        
        // Ensure canvas dimensions match video dimensions
        if (canvas.width !== video.videoWidth) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        // Only detect when video frame changes
        if (video.currentTime !== lastVideoTime) {
          lastVideoTime = video.currentTime;
          const results = detectPose(video, performance.now());

          if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
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
        }
      }
      requestRef.current = requestAnimationFrame(renderLoop);
    };

    if (isReady) {
      requestRef.current = requestAnimationFrame(renderLoop);
    }

    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [isReady, detectPose]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', backgroundColor: '#000' }}>
      {!isReady && <p style={{ color: 'white', zIndex: 10 }}>Loading High-Accuracy Pose Model...</p>}
      
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
