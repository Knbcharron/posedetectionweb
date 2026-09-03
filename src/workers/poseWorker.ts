import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

let poseLandmarker: PoseLandmarker | null = null;
let isReady = false;

// Initialize the landmarker when the worker receives INIT message
const initialize = async (options: any) => {
  try {
    const basePath = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
    
    // WORKAROUND for Vite module workers:
    // Module workers don't support importScripts, and eval() in strict mode isolates variables.
    // We wrap the legacy script in a Function to extract createMediaPipeModule and expose it globally.
    const response = await fetch(`${basePath}/vision_wasm_internal.js`);
    const scriptText = await response.text();
    const factory = new Function(`${scriptText}\nreturn ModuleFactory;`)();
    (self as any).ModuleFactory = factory;

    const vision = await FilesetResolver.forVisionTasks(basePath);

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "/pose_landmarker_full.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: options.numPoses ?? 1,
      minPoseDetectionConfidence: options.minPoseDetectionConfidence ?? 0.7,
      minPosePresenceConfidence: options.minPosePresenceConfidence ?? 0.7,
      minTrackingConfidence: options.minTrackingConfidence ?? 0.7,
    });
    
    isReady = true;
    postMessage({ type: "READY" });
  } catch (error) {
    console.error("Worker Initialization Error:", error);
  }
};

self.onmessage = (e: MessageEvent) => {
  const { type, image, timestamp, options } = e.data;

  if (type === "INIT") {
    initialize(options || {});
  } else if (type === "SET_OPTIONS" && poseLandmarker) {
    poseLandmarker.setOptions(options);
  } else if (type === "DETECT" && isReady && poseLandmarker) {
    try {
      const results = poseLandmarker.detectForVideo(image, timestamp);
      postMessage({ type: "RESULTS", results, timestamp });
    } catch (error) {
      console.error("Worker Detection Error:", error);
      postMessage({ type: "RESULTS", results: null, timestamp });
    } finally {
      // Close the ImageBitmap to free memory immediately
      if (image && typeof image.close === "function") {
        image.close();
      }
    }
  } else if (type === "CLOSE") {
    if (poseLandmarker) {
      poseLandmarker.close();
      poseLandmarker = null;
    }
    self.close();
  }
};
