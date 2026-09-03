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

  const [metrics, setMetrics] = useState({
    avgFps: 0,
    avgLatency: 0,
    dropRate: 0,
    successRate: 0,
    totalProcessed: 0
  });

  // Main rendering loop
  useEffect(() => {
    let lastVideoTime = -1;
    let drawingUtils: DrawingUtils | null = null;

    // Metrics state (cumulative)
    const stats = {
      startTime: 0,
      baseTotalFrames: 0,
      processedFrames: 0,
      successfulDetections: 0,
      latencies: [] as number[],
      lastUiUpdateTime: 0
    };

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

        const now = performance.now();
        
        // Initialize stats
        if (stats.startTime === 0) {
          stats.startTime = now;
          if (video.getVideoPlaybackQuality) {
            stats.baseTotalFrames = video.getVideoPlaybackQuality().totalVideoFrames;
          }
        }

        // Only detect when video frame changes
        if (video.currentTime !== lastVideoTime) {
          lastVideoTime = video.currentTime;
          
          const detectionStart = performance.now();
          const results = detectPose(video, detectionStart);
          const detectionEnd = performance.now();
          
          stats.processedFrames++;
          const latency = detectionEnd - detectionStart;
          stats.latencies.push(latency);

          if (results && results.landmarks && results.landmarks.length > 0) {
            stats.successfulDetections++;
          }

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

        // Update UI every 500ms
        if (now - stats.lastUiUpdateTime > 500) {
          stats.lastUiUpdateTime = now;
          
          let totalFrames = stats.processedFrames;
          if (video.getVideoPlaybackQuality) {
            totalFrames = Math.max(stats.processedFrames, video.getVideoPlaybackQuality().totalVideoFrames - stats.baseTotalFrames);
          }
          
          const droppedFrames = totalFrames - stats.processedFrames;
          const dropRate = totalFrames > 0 ? (droppedFrames / totalFrames) * 100 : 0;
          
          const durationSec = (now - stats.startTime) / 1000;
          const avgFps = durationSec > 0 ? stats.processedFrames / durationSec : 0;
          
          let avgLatency = 0;
          if (stats.latencies.length > 0) {
            const sum = stats.latencies.reduce((a, b) => a + b, 0);
            avgLatency = sum / stats.latencies.length;
          }

          const successRate = stats.processedFrames > 0 
            ? (stats.successfulDetections / stats.processedFrames) * 100 
            : 0;

          setMetrics({
            avgFps,
            avgLatency,
            dropRate,
            successRate,
            totalProcessed: stats.processedFrames
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
            <h3 style={{ margin: '0 0 10px 0', color: '#00ff00' }}>Performance Stats (Cumulative)</h3>
            <div style={{ margin: '5px 0' }}>Avg FPS: {metrics.avgFps.toFixed(1)}</div>
            <div style={{ margin: '5px 0' }}>Avg Latency (E2E): {metrics.avgLatency.toFixed(1)} ms</div>
            <div style={{ margin: '5px 0' }}>Frame Drop: {metrics.dropRate.toFixed(1)}%</div>
            <div style={{ margin: '5px 0' }}>Pose Detection Success: {metrics.successRate.toFixed(1)}%</div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
