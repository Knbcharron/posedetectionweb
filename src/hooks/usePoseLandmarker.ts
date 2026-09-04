import { useEffect, useRef, useState, useCallback } from "react";
import * as ort from "onnxruntime-web";

export interface YoloPoseResult {
  keypoints: { x: number; y: number; score: number }[];
  box: { x1: number; y1: number; x2: number; y2: number };
  score: number;
}

export const usePoseLandmarker = () => {
  const [isReady, setIsReady] = useState(false);
  const sessionRef = useRef<ort.InferenceSession | null>(null);

  // Pre-allocate arrays for performance
  const tensorInputRef = useRef<Float32Array>(new Float32Array(1 * 3 * 640 * 640));

  useEffect(() => {
    let active = true;

    const initModel = async () => {
      try {
        // Fix for pthread_create error: disable WASM multithreading
        // because it requires strict Cross-Origin isolation headers which dev servers lack.
        ort.env.wasm.numThreads = 1;
        
        // Try to use WebGL first for better performance, fallback to WASM (CPU)
        const session = await ort.InferenceSession.create("/yolov8n-pose.onnx", {
          executionProviders: ["webgl", "wasm"],
        });
        
        if (!active) return;
        
        sessionRef.current = session;
        setIsReady(true);
        console.log("YOLOv8 Pose Model Loaded!");
      } catch (error) {
        console.error("Error loading YOLO model:", error);
      }
    };

    initModel();

    return () => {
      active = false;
    };
  }, []);

  const detectPose = useCallback(async (
    videoElement: HTMLVideoElement,
    canvasForProcessing: HTMLCanvasElement
  ): Promise<YoloPoseResult[]> => {
    if (!sessionRef.current || !isReady) return [];

    const session = sessionRef.current;
    const ctx = canvasForProcessing.getContext("2d", { willReadFrequently: true });
    if (!ctx) return [];

    const inputSize = 640;
    
    // 1. Pre-processing: Letterboxing
    const w = videoElement.videoWidth;
    const h = videoElement.videoHeight;
    const scale = Math.min(inputSize / w, inputSize / h);
    const scaledW = Math.round(w * scale);
    const scaledH = Math.round(h * scale);
    const padX = (inputSize - scaledW) / 2;
    const padY = (inputSize - scaledH) / 2;

    canvasForProcessing.width = inputSize;
    canvasForProcessing.height = inputSize;
    
    // Fill with black (padding)
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, inputSize, inputSize);
    
    // Draw scaled video frame centered
    ctx.drawImage(videoElement, padX, padY, scaledW, scaledH);
    
    // 2. Extract Pixels & Normalize to NCHW format
    const imageData = ctx.getImageData(0, 0, inputSize, inputSize).data;
    const tensorData = tensorInputRef.current;
    
    // NCHW format (Batch=1, Channels=3, Height=640, Width=640)
    for (let i = 0; i < inputSize * inputSize; i++) {
      // Normalize [0, 255] to [0.0, 1.0]
      tensorData[i] = imageData[i * 4] / 255.0; // R
      tensorData[inputSize * inputSize + i] = imageData[i * 4 + 1] / 255.0; // G
      tensorData[inputSize * inputSize * 2 + i] = imageData[i * 4 + 2] / 255.0; // B
    }

    // 3. Inference
    const inputTensor = new ort.Tensor("float32", tensorData, [1, 3, inputSize, inputSize]);
    const results = await session.run({ images: inputTensor });
    
    // YOLOv8 Pose output shape: [1, 56, 8400]
    // 56 = 4 (box coords) + 1 (object score - actually not used directly like this in v8, it's max of class scores) + 51 (17 keypoints * 3 [x,y,conf])
    // Wait, YOLOv8 pose output is [1, 56, 8400] where 56 = 4 (cx,cy,w,h) + 1 (person class score) + 51 (keypoints)
    const output = results[session.outputNames[0]].data as Float32Array;
    
    // 4. Post-processing (NMS)
    const numDetections = 8400; // Typically 8400 for 640x640
    const numKeypoints = 17;
    
    let boxes: {x: number, y: number, w: number, h: number, score: number, index: number}[] = [];
    
    for (let i = 0; i < numDetections; i++) {
      // The output array is transposed: [56, 8400] flattened
      const cx = output[0 * numDetections + i];
      const cy = output[1 * numDetections + i];
      const w_box = output[2 * numDetections + i];
      const h_box = output[3 * numDetections + i];
      const score = output[4 * numDetections + i]; // Person class confidence
      
      if (score > 0.5) { // Confidence threshold
        boxes.push({
          x: cx - w_box / 2,
          y: cy - h_box / 2,
          w: w_box,
          h: h_box,
          score: score,
          index: i
        });
      }
    }
    
    // Simple NMS (Non-Maximum Suppression)
    boxes = boxes.sort((a, b) => b.score - a.score);
    const nmsThreshold = 0.45;
    const finalDetections: YoloPoseResult[] = [];
    
    while (boxes.length > 0) {
      const bestBox = boxes[0];
      
      // Extract keypoints for this detection
      const keypoints = [];
      const kptOffset = 5; // Start of keypoint data
      
      for (let k = 0; k < numKeypoints; k++) {
        const kx = output[(kptOffset + k * 3) * numDetections + bestBox.index];
        const ky = output[(kptOffset + k * 3 + 1) * numDetections + bestBox.index];
        const kscore = output[(kptOffset + k * 3 + 2) * numDetections + bestBox.index];
        
        // Scale keypoints back to original video dimensions
        keypoints.push({
          x: (kx - padX) / scale,
          y: (ky - padY) / scale,
          score: kscore
        });
      }
      
      finalDetections.push({
        box: {
          x1: (bestBox.x - padX) / scale,
          y1: (bestBox.y - padY) / scale,
          x2: (bestBox.x + bestBox.w - padX) / scale,
          y2: (bestBox.y + bestBox.h - padY) / scale
        },
        score: bestBox.score,
        keypoints
      });
      
      // Remove boxes with high IoU
      boxes = boxes.filter(box => {
        const intersectX1 = Math.max(bestBox.x, box.x);
        const intersectY1 = Math.max(bestBox.y, box.y);
        const intersectX2 = Math.min(bestBox.x + bestBox.w, box.x + box.w);
        const intersectY2 = Math.min(bestBox.y + bestBox.h, box.y + box.h);
        
        const intersectW = Math.max(0, intersectX2 - intersectX1);
        const intersectH = Math.max(0, intersectY2 - intersectY1);
        const intersectArea = intersectW * intersectH;
        
        const area1 = bestBox.w * bestBox.h;
        const area2 = box.w * box.h;
        const iou = intersectArea / (area1 + area2 - intersectArea);
        
        return iou < nmsThreshold;
      });
    }

    return finalDetections;
  }, [isReady]);

  return { detectPose, isReady };
};
