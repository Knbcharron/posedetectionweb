import { useEffect, useRef, useState } from "react";
import { usePoseLandmarker } from "./hooks/usePoseLandmarker";

// COCO Keypoint Skeleton Connections for YOLOv8
const SKELETON_CONNECTIONS = [
  [5, 6],   // Shoulders
  [5, 7],   // Left upper arm
  [7, 9],   // Left lower arm
  [6, 8],   // Right upper arm
  [8, 10],  // Right lower arm
  [5, 11],  // Left torso
  [6, 12],  // Right torso
  [11, 12], // Hips
  [11, 13], // Left upper leg
  [13, 15], // Left lower leg
  [12, 14], // Right upper leg
  [14, 16]  // Right lower leg
];

function App() {
  const { detectPose, isReady } = usePoseLandmarker();
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Hidden canvas used for extracting pixels for the ONNX model
  const processingCanvasRef = useRef<HTMLCanvasElement>(null); 
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

  const [metrics, setMetrics] = useState({
    avgFps: 0,
    avgLatency: 0,
    totalProcessed: 0,
    successfulDetections: 0
  });

  // Main rendering loop
  useEffect(() => {
    let lastVideoTime = -1;

    // Metrics state (cumulative)
    const stats = {
      startTime: 0,
      processedFrames: 0,
      successfulDetections: 0,
      latencies: [] as number[],
      lastUiUpdateTime: 0
    };

    const renderLoop = async () => {
      if (
        isReady &&
        videoRef.current &&
        videoRef.current.readyState >= 2 &&
        canvasRef.current &&
        processingCanvasRef.current
      ) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        
        // Ensure canvas dimensions match video dimensions
        if (canvas.width !== video.videoWidth) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        const now = performance.now();
        
        // Initialize stats
        if (stats.startTime === 0) {
          stats.startTime = now;
        }

        // Only detect when video frame changes
        if (video.currentTime !== lastVideoTime) {
          lastVideoTime = video.currentTime;
          
          const detectionStart = performance.now();
          // detectPose now takes the video and the processing canvas
          const results = await detectPose(video, processingCanvasRef.current);
          const detectionEnd = performance.now();
          
          stats.processedFrames++;
          const latency = detectionEnd - detectionStart;
          stats.latencies.push(latency);

          if (results && results.length > 0) {
            stats.successfulDetections++;
          }

          if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw skeleton if landmarks are detected
            if (results && results.length > 0) {
              for (const person of results) {
                // Draw Connections
                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = 3;
                ctx.beginPath();
                
                // Add facial connections just in case
                const ALL_CONNECTIONS = [
                  ...SKELETON_CONNECTIONS,
                  [0, 1], [1, 3], // Left face
                  [0, 2], [2, 4]  // Right face
                ];

                for (const [idx1, idx2] of ALL_CONNECTIONS) {
                  const pt1 = person.keypoints[idx1];
                  const pt2 = person.keypoints[idx2];
                  
                  // YOLOv8 keypoint confidences might be unnormalized or lower than expected.
                  // Let's lower the threshold so the skeleton is more likely to draw.
                  if (pt1 && pt2 && pt1.score > 0.1 && pt2.score > 0.1) {
                    ctx.moveTo(pt1.x, pt1.y);
                    ctx.lineTo(pt2.x, pt2.y);
                  }
                }
                ctx.stroke();

                // Draw Keypoints
                ctx.fillStyle = "#FF0000";
                for (let i = 0; i < person.keypoints.length; i++) {
                  const pt = person.keypoints[i];
                  if (pt.score > 0.1) {
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, 4, 0, 2 * Math.PI);
                    ctx.fill();
                  }
                }
              }
            }
          }
        }

        // Update UI every 500ms
        if (now - stats.lastUiUpdateTime > 500) {
          stats.lastUiUpdateTime = now;
          
          const durationSec = (now - stats.startTime) / 1000;
          const avgFps = durationSec > 0 ? stats.processedFrames / durationSec : 0;
          
          let avgLatency = 0;
          if (stats.latencies.length > 0) {
            const sum = stats.latencies.reduce((a, b) => a + b, 0);
            avgLatency = sum / stats.latencies.length;
          }

          setMetrics({
            avgFps,
            avgLatency,
            totalProcessed: stats.processedFrames,
            successfulDetections: stats.successfulDetections
          });
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
      {!isReady && <p style={{ color: 'white', zIndex: 10 }}>Loading YOLOv8 Pose Model (WASM)...</p>}
      
      {hasCameraPermission === false && (
        <p style={{ color: 'red', zIndex: 10 }}>Camera access denied. Please grant permissions to use this app.</p>
      )}

      {/* Hidden canvas for extracting tensors */}
      <canvas ref={processingCanvasRef} style={{ display: 'none' }} />

      <div style={{ position: 'absolute', top: 0, left: 0, width: '100vw', height: '100vh', overflow: 'hidden', backgroundColor: 'black' }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
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
        
        {/* Metrics Overlay */}
        {isReady && metrics.totalProcessed > 0 && (
          <div style={{
            position: 'absolute',
            top: 20,
            left: 20,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            padding: '15px',
            borderRadius: '8px',
            color: 'white',
            fontFamily: 'monospace',
            zIndex: 20,
            pointerEvents: 'none',
            textShadow: '1px 1px 2px black'
          }}>
            <h3 style={{ margin: '0 0 10px 0', color: '#00ff00' }}>YOLOv8 Performance Stats</h3>
            <div style={{ margin: '5px 0' }}>Avg FPS: {metrics.avgFps.toFixed(1)}</div>
            <div style={{ margin: '5px 0' }}>Avg Latency (E2E): {metrics.avgLatency.toFixed(1)} ms</div>
            <div style={{ margin: '5px 0' }}>Frames Processed: {metrics.totalProcessed}</div>
            <div style={{ margin: '5px 0' }}>Detections: {metrics.successfulDetections}</div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
