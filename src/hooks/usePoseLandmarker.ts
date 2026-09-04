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
  const prevDetectionsRef = useRef<YoloPoseResult[]>([]);

  // Pre-allocate arrays for performance
  const tensorInputRef = useRef<Float32Array>(new Float32Array(1 * 3 * 640 * 640));

  useEffect(() => {
    let active = true;

    const initModel = async () => {
      try {
        // Fix for pthread_create error: disable WASM multithreading
        // because it requires strict Cross-Origin isolation headers which dev servers lack.
        ort.env.wasm.numThreads = 1;
        
        // Try to use WebGPU first (fastest), then WebGL, fallback to WASM (CPU)
        const session = await ort.InferenceSession.create("/yolov8n-pose.onnx", {
          executionProviders: ["webgpu", "webgl", "wasm"],
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
    
    // Check and resize input tensor array if needed (dynamic safety)
    if (tensorInputRef.current.length !== 1 * 3 * inputSize * inputSize) {
      tensorInputRef.current = new Float32Array(1 * 3 * inputSize * inputSize);
    }
    
    // 1. Pre-processing: Letterboxing
    const w = videoElement.videoWidth;
    const h = videoElement.videoHeight;
    if (w === 0 || h === 0) return []; // Guard against zero-dim video early
    
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
      tensorData[i] = imageData[i * 4] / 255.0; // R
      tensorData[inputSize * inputSize + i] = imageData[i * 4 + 1] / 255.0; // G
      tensorData[inputSize * inputSize * 2 + i] = imageData[i * 4 + 2] / 255.0; // B
    }

    // 3. Inference
    const inputTensor = new ort.Tensor("float32", tensorData, [1, 3, inputSize, inputSize]);
    const results = await session.run({ images: inputTensor });
    
    const outputTensor = results[session.outputNames[0]];
    const output = outputTensor.data as Float32Array;
    const outputDims = outputTensor.dims;
    
    // 4. Post-processing (Dynamic parsing to handle all YOLOv8 export formats)
    let numDetections = 0;
    let numFeatures = 0;
    let isTransposed = false;
    
    // Check if output is [1, 56, 8400] or [1, 8400, 56]
    if (outputDims[1] > outputDims[2]) {
      numDetections = outputDims[1];
      numFeatures = outputDims[2];
      isTransposed = true; // [1, 8400, 56] format
    } else {
      numDetections = outputDims[2];
      numFeatures = outputDims[1];
      isTransposed = false; // [1, 56, 8400] format
    }
    
    const numKeypoints = (numFeatures - 5) / 3;
    
    let boxes: {x: number, y: number, w: number, h: number, score: number, index: number}[] = [];
    
    for (let i = 0; i < numDetections; i++) {
      // Dynamic index extraction based on transposition
      const cx = isTransposed ? output[i * numFeatures + 0] : output[0 * numDetections + i];
      const cy = isTransposed ? output[i * numFeatures + 1] : output[1 * numDetections + i];
      const w_box = isTransposed ? output[i * numFeatures + 2] : output[2 * numDetections + i];
      const h_box = isTransposed ? output[i * numFeatures + 3] : output[3 * numDetections + i];
      const score = isTransposed ? output[i * numFeatures + 4] : output[4 * numDetections + i];
      
      if (score > 0.5) { 
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
        const kx = isTransposed 
          ? output[bestBox.index * numFeatures + (kptOffset + k * 3)] 
          : output[(kptOffset + k * 3) * numDetections + bestBox.index];
        const ky = isTransposed 
          ? output[bestBox.index * numFeatures + (kptOffset + k * 3 + 1)] 
          : output[(kptOffset + k * 3 + 1) * numDetections + bestBox.index];
        const kscore = isTransposed 
          ? output[bestBox.index * numFeatures + (kptOffset + k * 3 + 2)] 
          : output[(kptOffset + k * 3 + 2) * numDetections + bestBox.index];
        
        // Scale keypoints back to original video dimensions exactly mapping
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

    // 5. Temporal Smoothing (EMA) to reduce jitter
    const SMOOTHING = 0.6; // Higher = more smoothing (less jitter, more lag)
    
    // Match current detections with previous detections based on Box IoU
    for (let i = 0; i < finalDetections.length; i++) {
      const curr = finalDetections[i];
      let bestPrevIdx = -1;
      let bestIoU = 0;
      
      for (let j = 0; j < prevDetectionsRef.current.length; j++) {
        const prev = prevDetectionsRef.current[j];
        
        // Calculate IoU
        const intersectX1 = Math.max(curr.box.x1, prev.box.x1);
        const intersectY1 = Math.max(curr.box.y1, prev.box.y1);
        const intersectX2 = Math.min(curr.box.x2, prev.box.x2);
        const intersectY2 = Math.min(curr.box.y2, prev.box.y2);
        
        const intersectW = Math.max(0, intersectX2 - intersectX1);
        const intersectH = Math.max(0, intersectY2 - intersectY1);
        const intersectArea = intersectW * intersectH;
        
        const area1 = (curr.box.x2 - curr.box.x1) * (curr.box.y2 - curr.box.y1);
        const area2 = (prev.box.x2 - prev.box.x1) * (prev.box.y2 - prev.box.y1);
        const iou = intersectArea / (area1 + area2 - intersectArea);
        
        if (iou > bestIoU) {
          bestIoU = iou;
          bestPrevIdx = j;
        }
      }
      
      // If we found a match (IoU > 0.3), apply EMA smoothing
      if (bestIoU > 0.3) {
        const prev = prevDetectionsRef.current[bestPrevIdx];
        
        // Smooth Keypoints
        for (let k = 0; k < curr.keypoints.length; k++) {
          curr.keypoints[k].x = curr.keypoints[k].x * (1 - SMOOTHING) + prev.keypoints[k].x * SMOOTHING;
          curr.keypoints[k].y = curr.keypoints[k].y * (1 - SMOOTHING) + prev.keypoints[k].y * SMOOTHING;
          curr.keypoints[k].score = curr.keypoints[k].score * (1 - SMOOTHING) + prev.keypoints[k].score * SMOOTHING;
        }
        
        // Smooth Bounding Box
        curr.box.x1 = curr.box.x1 * (1 - SMOOTHING) + prev.box.x1 * SMOOTHING;
        curr.box.y1 = curr.box.y1 * (1 - SMOOTHING) + prev.box.y1 * SMOOTHING;
        curr.box.x2 = curr.box.x2 * (1 - SMOOTHING) + prev.box.x2 * SMOOTHING;
        curr.box.y2 = curr.box.y2 * (1 - SMOOTHING) + prev.box.y2 * SMOOTHING;
      }
    }
    
    // Save current detections for next frame
    prevDetectionsRef.current = JSON.parse(JSON.stringify(finalDetections));

    return finalDetections;
  }, [isReady]);

  return { detectPose, isReady };
};
