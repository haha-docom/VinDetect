/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Arrays;
import java.util.Vector;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * - https://github.com/tensorflow/models/tree/master/research/object_detection
 * where you can find the training code.
 *
 * To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
public class TFLiteObjectDetectionAPIModel implements Classifier {
  private static final Logger LOGGER = new Logger();

  // Only return this many results.
  // New model returns up to 50 detections
  private static final int NUM_DETECTIONS = 50;
  // Float model
  private static final float IMAGE_MEAN = 127.5f;
  private static final float IMAGE_STD = 127.5f;
  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize;
  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][] outputLocations;
  // If the model's output is quantized (e.g. UINT8) we keep the raw buffer here
  private Object outputLocationsRaw = null;
  private float outputLocationsScale = 0f;
  private int outputLocationsZeroPoint = 0;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][] outputClasses;
  private Object outputClassesRaw = null;
  private float outputClassesScale = 0f;
  private int outputClassesZeroPoint = 0;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][] outputScores;
  private Object outputScoresRaw = null;
  private float outputScoresScale = 0f;
  private int outputScoresZeroPoint = 0;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes
  private float[] numDetections;
  private Object numDetectionsRaw = null;
  private float numDetectionsScale = 0f;
  private int numDetectionsZeroPoint = 0;

  private ByteBuffer imgData;

  private Interpreter tfLite;
  private MappedByteBuffer modelFileBuffer;
  private ProcessorType currentProcessorType = ProcessorType.CPU;

  // Enable a full dump of output tensors to logcat. Set to false in production.
  private static final boolean FULL_DUMP_OUTPUTS = true;

  // Output tensor index mapping (by tensor name) for robust handling of models that
  // return a dict / named outputs.
  private int idxLocations = -1;
  private int idxClasses = -1;
  private int idxScores = -1;
  private int idxNumDetections = -1;
  // Some models (raw SSD) return logits/probabilities with shape [1, anchors, num_classes]
  // e.g. [1,12804,2] -> logits for background/foreground per anchor.
  private int idxLogits = -1;
  private float[][][] outputLogits = null; // [1, anchors, num_classes]
  private Object outputLogitsRaw = null;
  private float outputLogitsScale = 0f;
  private int outputLogitsZeroPoint = 0;
  private Map<String, Integer> outputIndexMap = new HashMap<>();

  private TFLiteObjectDetectionAPIModel() {}

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static Classifier create(
      final AssetManager assetManager,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final boolean isQuantized,
      final ProcessorType processType
      )
      throws IOException {
    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    InputStream labelsInput = assetManager.open(actualFilename);
    BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      LOGGER.w(line);
      d.labels.add(line);
    }
    br.close();

    d.inputSize = inputSize;

  Interpreter.Options options = new Interpreter.Options();
    try {
      switch (processType) {
        case CPU:   // cpu
          Log.i("LUONG", "Đang dùng cpu");
          break;
        case NPU:   // npu
          NnApiDelegate nnApiDelegate = new NnApiDelegate(); // Khởi tạo NNAPI delegate
          options.addDelegate(nnApiDelegate);
          Log.i("LUONG", "Đang dùng NNAPI, đăng kí delegate NNAPI delegate created: " + (nnApiDelegate != null));
          break;
        case GPU: // gpu
          Log.i("LUONG", "Đang dùng gpu");
          GpuDelegate delegate = new GpuDelegate();
          options.addDelegate(delegate);
          break;
        default:
          // Default to CPU (no delegate) to avoid unexpected NNAPI usage on devices
          Log.i("LUONG", "ProcessorType default: using CPU (no delegate)");
          break;
      }
      // Load model file into a buffer and keep it so we can recreate the interpreter at runtime
      MappedByteBuffer modelBuffer = loadModelFile(assetManager, modelFilename);
      d.modelFileBuffer = modelBuffer;
      d.currentProcessorType = processType;
      // Ensure thread count is set on options (Interpreter.setNumThreads removed in newer TF versions)
      try {
        options.setNumThreads(NUM_THREADS);
      } catch (Exception ignored) {}
      d.tfLite = new Interpreter(d.modelFileBuffer, options);

    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputSize * d.inputSize];

  // set threads via options before creating interpreter
  // (Interpreter.setNumThreads was removed in newer TF Lite versions)
  // options will be created/configured above; ensure default thread count
  // was already applied to options earlier.
    // Discover output tensor names and map them to indices so callers can use
    // named outputs (detection_scores, detection_boxes, detection_classes, num_detections).
    try {
      int outCount = d.tfLite.getOutputTensorCount();
      for (int i = 0; i < outCount; i++) {
        String outName = d.tfLite.getOutputTensor(i).name();
        d.outputIndexMap.put(outName, i);
        int[] shape = d.tfLite.getOutputTensor(i).shape();
        DataType dt = d.tfLite.getOutputTensor(i).dataType();
        Log.i("TFLiteModel", "Output tensor " + i + " name=" + outName + " shape=" + java.util.Arrays.toString(shape) + " dtype=" + dt);

        // Shape-based mapping (more robust than relying on names/order)
        if (shape != null) {
          if (shape.length == 3 && shape[0] == 1 && shape[2] == 4) {
            // [1, N, 4] => boxes
            d.idxLocations = i;
            continue;
          }
          // common SSD logits/prob output shape: [1, anchors, classes]
          if (shape.length == 3 && shape[0] == 1 && shape[2] > 1) {
            // treat as logits/probabilities per-anchor
            d.idxLogits = i;
            continue;
          }
          if (shape.length == 2 && shape[0] == 1) {
            // [1, N] => either scores or classes. Many exported models use N != NUM_DETECTIONS
            // (e.g. 25). Decide by name if possible, otherwise assign to the first unassigned
            // (prefer scores then classes). Don't rely on dtype (UINT8 is a valid quantized scores tensor).
            String nlow = outName.toLowerCase();
            if (nlow.contains("score") || nlow.contains("scores") || nlow.contains("detection_scores")) {
              d.idxScores = i;
              continue;
            } else if (nlow.contains("class") || nlow.contains("classes") || nlow.contains("detection_classes")) {
              d.idxClasses = i;
              continue;
            }
            // fallback: prefer assigning to scores if free, else classes
            if (d.idxScores == -1) {
              d.idxScores = i;
              continue;
            }
            if (d.idxClasses == -1) {
              d.idxClasses = i;
              continue;
            }
          }
          // num_detections often has shape [1] or [1,1]
          if ((shape.length == 1 && shape[0] == 1) || (shape.length == 2 && shape[0] == 1 && shape[1] == 1)) {
            d.idxNumDetections = i;
            continue;
          }
        }
        // Name-based fallback
        if (outName.toLowerCase().contains("detection_boxes")) d.idxLocations = i;
        if (outName.toLowerCase().contains("detection_classes")) d.idxClasses = i;
        if (outName.toLowerCase().contains("detection_scores")) d.idxScores = i;
        if (outName.toLowerCase().contains("num_detections") || outName.toLowerCase().contains("num_detections")) d.idxNumDetections = i;
      }
    } catch (Exception e) {
      Log.i("TFLiteModel", "Could not inspect output tensors by name/shape: " + e.getMessage());
    }
    // Allocate output buffers based on discovered output tensor shapes when possible.
    try {
      if (d.idxLocations != -1) {
        int[] s = d.tfLite.getOutputTensor(d.idxLocations).shape();
        // shape expected [batch, num, 4]
        DataType dt = d.tfLite.getOutputTensor(d.idxLocations).dataType();
        if (dt == DataType.UINT8) {
          // allocate raw byte buffer matching tensor shape
          d.outputLocationsRaw = new byte[s[0]][s[1]][s[2]];
          try {
            d.outputLocationsScale = d.tfLite.getOutputTensor(d.idxLocations).quantizationParams().getScale();
            d.outputLocationsZeroPoint = d.tfLite.getOutputTensor(d.idxLocations).quantizationParams().getZeroPoint();
          } catch (Exception ignored) {}
        } else {
          d.outputLocations = new float[s[0]][s[1]][s[2]];
        }
      }
      if (d.idxLogits != -1) {
        int[] s = d.tfLite.getOutputTensor(d.idxLogits).shape();
        DataType dt = d.tfLite.getOutputTensor(d.idxLogits).dataType();
        if (dt == DataType.UINT8) {
          d.outputLogitsRaw = new byte[s[0]][s[1]][s[2]];
          try {
            d.outputLogitsScale = d.tfLite.getOutputTensor(d.idxLogits).quantizationParams().getScale();
            d.outputLogitsZeroPoint = d.tfLite.getOutputTensor(d.idxLogits).quantizationParams().getZeroPoint();
          } catch (Exception ignored) {}
        } else {
          d.outputLogits = new float[s[0]][s[1]][s[2]];
        }
      }
      if (d.idxScores != -1) {
        int[] s = d.tfLite.getOutputTensor(d.idxScores).shape();
        DataType dt = d.tfLite.getOutputTensor(d.idxScores).dataType();
        if (dt == DataType.UINT8) {
          d.outputScoresRaw = new byte[s[0]][s[1]];
          try {
            d.outputScoresScale = d.tfLite.getOutputTensor(d.idxScores).quantizationParams().getScale();
            d.outputScoresZeroPoint = d.tfLite.getOutputTensor(d.idxScores).quantizationParams().getZeroPoint();
          } catch (Exception ignored) {}
        } else {
          d.outputScores = new float[s[0]][s[1]];
        }
      }
      if (d.idxClasses != -1) {
        int[] s = d.tfLite.getOutputTensor(d.idxClasses).shape();
        DataType dt = d.tfLite.getOutputTensor(d.idxClasses).dataType();
        if (dt == DataType.UINT8) {
          d.outputClassesRaw = new byte[s[0]][s[1]];
          try {
            d.outputClassesScale = d.tfLite.getOutputTensor(d.idxClasses).quantizationParams().getScale();
            d.outputClassesZeroPoint = d.tfLite.getOutputTensor(d.idxClasses).quantizationParams().getZeroPoint();
          } catch (Exception ignored) {}
        } else {
          d.outputClasses = new float[s[0]][s[1]];
        }
      }
      if (d.idxNumDetections != -1) {
        int[] s = d.tfLite.getOutputTensor(d.idxNumDetections).shape();
        DataType dt = d.tfLite.getOutputTensor(d.idxNumDetections).dataType();
        if (dt == DataType.UINT8) {
          // numDetections often shape [1] or [1,1]
          if (s.length == 1) d.numDetectionsRaw = new byte[s[0]];
          else if (s.length == 2) d.numDetectionsRaw = new byte[s[0]][s[1]];
          try {
            d.numDetectionsScale = d.tfLite.getOutputTensor(d.idxNumDetections).quantizationParams().getScale();
            d.numDetectionsZeroPoint = d.tfLite.getOutputTensor(d.idxNumDetections).quantizationParams().getZeroPoint();
          } catch (Exception ignored) {}
        } else {
          d.numDetections = new float[s[0]];
        }
      }
    } catch (Exception e) {
      // Fallback to conservative defaults if shape inspection fails
      d.outputLocations = new float[1][NUM_DETECTIONS][4];
      d.outputClasses = new float[1][NUM_DETECTIONS];
      d.outputScores = new float[1][NUM_DETECTIONS];
      d.numDetections = new float[1];
    }
    return d;
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];
        if (isModelQuantized) {
          // Quantized model
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        } else { // Float model
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        }
      }
    }
    Trace.endSection(); // preprocessBitmap

  // Copy the input data into TensorFlow.
  Trace.beginSection("feed");
  // Allocate conservative defaults only if neither raw nor float buffers were set.
  if (outputLocations == null && outputLocationsRaw == null) outputLocations = new float[1][NUM_DETECTIONS][4];
  if (outputClasses == null && outputClassesRaw == null) outputClasses = new float[1][NUM_DETECTIONS];
  if (outputScores == null && outputScoresRaw == null) outputScores = new float[1][NUM_DETECTIONS];
  if (numDetections == null && numDetectionsRaw == null) numDetections = new float[1];

    Object[] inputArray = {imgData};
    Map<Integer, Object> outputMap = new HashMap<>();

    // Use discovered indices when available, otherwise fall back to typical ordering.
    // Put the appropriate buffer type (raw byte buffer for UINT8, float buffer otherwise)
    if (idxLocations != -1) {
      outputMap.put(idxLocations, outputLocationsRaw != null ? outputLocationsRaw : outputLocations);
    }
    if (idxClasses != -1) {
      outputMap.put(idxClasses, outputClassesRaw != null ? outputClassesRaw : outputClasses);
    }
    if (idxScores != -1) {
      outputMap.put(idxScores, outputScoresRaw != null ? outputScoresRaw : outputScores);
    }
    if (idxNumDetections != -1) {
      outputMap.put(idxNumDetections, numDetectionsRaw != null ? numDetectionsRaw : numDetections);
    }

    // Fallback: if none of the named indices were detected, assume stable ordering
    if (outputMap.isEmpty()) {
      outputMap.put(0, outputLocations);
      outputMap.put(1, outputClasses);
      outputMap.put(2, outputScores);
      outputMap.put(3, numDetections);
    }
    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("run");
    // Ensure logits buffer is included in output map if present so interpreter will fill it
    if (idxLogits != -1) {
      outputMap.put(idxLogits, outputLogitsRaw != null ? outputLogitsRaw : outputLogits);
    }
    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
    // Diagnostic: log output tensor dtypes and quant params, and a small raw-byte sample
    try {
      LOGGER.i("POST-RUN diagnostics:");
      for (Map.Entry<String, Integer> e : outputIndexMap.entrySet()) {
        String name = e.getKey();
        int idx = e.getValue();
        try {
          DataType dt = tfLite.getOutputTensor(idx).dataType();
          float qs = 0f; int qz = 0;
          try {
            qs = tfLite.getOutputTensor(idx).quantizationParams().getScale();
            qz = tfLite.getOutputTensor(idx).quantizationParams().getZeroPoint();
          } catch (Exception ignored) {}
          LOGGER.i("OUT postrun name=" + name + " idx=" + idx + " dtype=" + dt + " scale=" + qs + " zeroPoint=" + qz);
        } catch (Exception ex) {
          LOGGER.i("OUT postrun name=" + name + " idx=" + idx + " error=" + ex.getMessage());
        }
      }
      // If raw buffers were allocated, print a small raw sample before dequant to ensure interpreter wrote bytes
      if (outputScoresRaw != null) {
        try {
          byte[][] raw = (byte[][]) outputScoresRaw;
          int n = raw[0].length;
          int sample = Math.min(8, n);
          StringBuilder sb = new StringBuilder();
          sb.append("RAW_SCORE_BYTES[0][0.." + sample + "]=\n[");
          for (int i = 0; i < sample; ++i) { sb.append(raw[0][i] & 0xFF); if (i < sample-1) sb.append(','); }
          sb.append(']');
          LOGGER.i(sb.toString());
        } catch (ClassCastException ignored) { LOGGER.i("RAW_SCORE_BYTES: cast error"); }
      }
      if (outputClassesRaw != null) {
        try {
          byte[][] raw = (byte[][]) outputClassesRaw;
          int n = raw[0].length;
          int sample = Math.min(8, n);
          StringBuilder sb = new StringBuilder();
          sb.append("RAW_CLASS_BYTES[0][0.." + sample + "]=\n[");
          for (int i = 0; i < sample; ++i) { sb.append(raw[0][i] & 0xFF); if (i < sample-1) sb.append(','); }
          sb.append(']');
          LOGGER.i(sb.toString());
        } catch (ClassCastException ignored) { LOGGER.i("RAW_CLASS_BYTES: cast error"); }
      }
      if (outputLocationsRaw != null) {
        try {
          byte[][][] raw = (byte[][][]) outputLocationsRaw;
          if (raw[0].length > 0 && raw[0][0].length >= 4) {
            StringBuilder sb = new StringBuilder();
            sb.append("RAW_BOX_BYTES[0][0][0..3]=[");
            for (int i = 0; i < 4; ++i) { sb.append(raw[0][0][i] & 0xFF); if (i < 3) sb.append(','); }
            sb.append(']');
            LOGGER.i(sb.toString());
          }
        } catch (ClassCastException ignored) { LOGGER.i("RAW_BOX_BYTES: cast error"); }
      }
      if (numDetectionsRaw != null) {
        try {
          if (numDetectionsRaw instanceof byte[]) {
            byte[] raw = (byte[]) numDetectionsRaw;
            LOGGER.i("RAW_NUMDETS_BYTE[0]=" + (raw.length>0 ? (raw[0] & 0xFF) : -1));
          } else if (numDetectionsRaw instanceof byte[][]) {
            byte[][] raw = (byte[][]) numDetectionsRaw;
            LOGGER.i("RAW_NUMDETS_BYTE[0][0]=" + (raw.length>0 && raw[0].length>0 ? (raw[0][0] & 0xFF) : -1));
          }
        } catch (Exception ignored) { LOGGER.i("RAW_NUMDETS: read error"); }
      }
    } catch (Exception ignored) {}
    // Diagnostic: always log whether raw buffers exist and current quant params
    try {
      LOGGER.i("DIAG_FLAGS outputScoresRawNull=" + (outputScoresRaw == null) + " outputClassesRawNull=" + (outputClassesRaw == null) + " outputLocationsRawNull=" + (outputLocationsRaw == null) + " numDetectionsRawNull=" + (numDetectionsRaw == null));
      LOGGER.i("DIAG_QP scores_scale=" + outputScoresScale + " scores_zp=" + outputScoresZeroPoint + " classes_scale=" + outputClassesScale + " classes_zp=" + outputClassesZeroPoint);
      if (outputScores != null && outputScores.length > 0) {
        int show = Math.min(8, outputScores[0].length);
        LOGGER.i("DIAG_SCOREFLOAT sample=" + Arrays.toString(Arrays.copyOfRange(outputScores[0], 0, show)));
      } else {
        LOGGER.i("DIAG_SCOREFLOAT sample=null");
      }
    } catch (Exception ignored) {}

    // If any outputs were quantized (UINT8) we need to dequantize into float arrays
    try {
      // dequantize locations
      if (outputLocationsRaw != null) {
        try {
          byte[][][] raw = (byte[][][]) outputLocationsRaw;
          int b0 = raw.length;
          int a = raw[0].length;
          int c = raw[0][0].length;
          outputLocations = new float[b0][a][c];
          for (int bi = 0; bi < b0; ++bi) {
            for (int ai = 0; ai < a; ++ai) {
              for (int ci = 0; ci < c; ++ci) {
                int unsigned = raw[bi][ai][ci] & 0xFF;
                outputLocations[bi][ai][ci] = (unsigned - outputLocationsZeroPoint) * outputLocationsScale;
              }
            }
          }
        } catch (ClassCastException ignored) {}
      }
      // dequantize logits
      if (outputLogitsRaw != null) {
        try {
          byte[][][] raw = (byte[][][]) outputLogitsRaw;
          int b0 = raw.length;
          int anchors = raw[0].length;
          int classes = raw[0][0].length;
          outputLogits = new float[b0][anchors][classes];
          for (int bi = 0; bi < b0; ++bi) {
            for (int ai = 0; ai < anchors; ++ai) {
              for (int ci = 0; ci < classes; ++ci) {
                int unsigned = raw[bi][ai][ci] & 0xFF;
                outputLogits[bi][ai][ci] = (unsigned - outputLogitsZeroPoint) * outputLogitsScale;
              }
            }
          }
        } catch (ClassCastException ignored) {}
      }
      // dequantize scores
      if (outputScoresRaw != null) {
        try {
          byte[][] raw = (byte[][]) outputScoresRaw;
          int b0 = raw.length;
          int n = raw[0].length;
          outputScores = new float[b0][n];
          for (int bi = 0; bi < b0; ++bi) {
            for (int i = 0; i < n; ++i) {
              int unsigned = raw[bi][i] & 0xFF;
              outputScores[bi][i] = (unsigned - outputScoresZeroPoint) * outputScoresScale;
            }
          }
          // Diagnostic logging: quantization params and simple statistics for scores
          try {
            // Log quantization params so we can verify dequant formula
            LOGGER.i("SCORES_QUANT scale=" + outputScoresScale + " zeroPoint=" + outputScoresZeroPoint);
            // Log a small raw sample (first 8 anchors)
            int sampleLen = Math.min(8, n);
            StringBuilder rawSampleSb = new StringBuilder();
            rawSampleSb.append("raw[0][0.." + sampleLen + "]=\n[");
            for (int si = 0; si < sampleLen; ++si) {
              rawSampleSb.append((raw[0][si] & 0xFF));
              if (si < sampleLen - 1) rawSampleSb.append(',');
            }
            rawSampleSb.append(']');
            LOGGER.i(rawSampleSb.toString());

            // Compute min/max/mean of dequantized scores for the first batch
            float min = Float.POSITIVE_INFINITY;
            float max = Float.NEGATIVE_INFINITY;
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
              float v = outputScores[0][i];
              if (v < min) min = v;
              if (v > max) max = v;
              sum += v;
            }
            double mean = n > 0 ? sum / n : 0.0;
            LOGGER.i("SCORES_STATS min=" + min + " max=" + max + " mean=" + mean + " count=" + n);
          } catch (Exception ex) {
            LOGGER.i("SCORES_STATS error: " + ex.getMessage());
          }
        } catch (ClassCastException ignored) {}
      }
      // dequantize classes
      if (outputClassesRaw != null) {
        try {
          byte[][] raw = (byte[][]) outputClassesRaw;
          int b0 = raw.length;
          int n = raw[0].length;
          outputClasses = new float[b0][n];
          for (int bi = 0; bi < b0; ++bi) {
            for (int i = 0; i < n; ++i) {
              int unsigned = raw[bi][i] & 0xFF;
              outputClasses[bi][i] = (unsigned - outputClassesZeroPoint) * outputClassesScale;
            }
          }
        } catch (ClassCastException ignored) {}
      }
      // dequantize numDetections
      if (numDetectionsRaw != null) {
        try {
          if (numDetectionsRaw instanceof byte[]) {
            byte[] raw = (byte[]) numDetectionsRaw;
            numDetections = new float[raw.length];
            for (int i = 0; i < raw.length; ++i) {
              int unsigned = raw[i] & 0xFF;
              numDetections[i] = (unsigned - numDetectionsZeroPoint) * numDetectionsScale;
            }
          } else if (numDetectionsRaw instanceof byte[][]) {
            byte[][] raw = (byte[][]) numDetectionsRaw;
            int b0 = raw.length;
            numDetections = new float[b0];
            for (int bi = 0; bi < b0; ++bi) {
              int unsigned = raw[bi][0] & 0xFF;
              numDetections[bi] = (unsigned - numDetectionsZeroPoint) * numDetectionsScale;
            }
          }
        } catch (ClassCastException ignored) {}
      }
    } catch (Exception ignored) {}
  // Log outputs for debugging to verify interpreter populated all expected tensors.
  LOGGER.i("Model numDetections (raw): " + (numDetections != null && numDetections.length > 0 ? numDetections[0] : -1));
  LOGGER.i("Model first output score: " + (outputScores != null && outputScores.length > 0 && outputScores[0].length > 0 ? outputScores[0][0] : -1f));

  // Additional debug: print a map of all discovered output tensors and a small sample of their values.
  try {
    for (Map.Entry<String, Integer> e : outputIndexMap.entrySet()) {
      String name = e.getKey();
      int idx = e.getValue();
      int[] shape = tfLite.getOutputTensor(idx).shape();
      String sample = "";
      // Print a few representative values depending on shape/known names
      if (name.toLowerCase().contains("score") && outputScores != null) {
        sample = Arrays.toString(java.util.Arrays.copyOfRange(outputScores[0], 0, Math.min(8, outputScores[0].length)));
      } else if (name.toLowerCase().contains("class") && outputClasses != null) {
        sample = Arrays.toString(java.util.Arrays.copyOfRange(outputClasses[0], 0, Math.min(8, outputClasses[0].length)));
      } else if (name.toLowerCase().contains("box") && outputLocations != null) {
        // flatten and show first box
        if (outputLocations[0].length > 0) {
          sample = Arrays.toString(outputLocations[0][0]);
        }
      } else if (name.toLowerCase().contains("num") && numDetections != null) {
        sample = Arrays.toString(numDetections);
      }
      LOGGER.i("Output tensor: " + name + " idx=" + idx + " shape=" + java.util.Arrays.toString(shape) + " sample=" + sample);
    }
  } catch (Exception ignored) {}
    Trace.endSection();

    // Full dump of output tensors to logcat (verbose). Controlled by FULL_DUMP_OUTPUTS.
    if (FULL_DUMP_OUTPUTS) {
      try {
        int fullCount = 0;
        if (numDetections != null && numDetections.length > 0) {
          fullCount = Math.min(NUM_DETECTIONS, (int) numDetections[0]);
        }
        LOGGER.i("FULL_DUMP numDetections(raw)=" + (numDetections != null && numDetections.length > 0 ? numDetections[0] : -1) + " count=" + fullCount);

        if (outputScores != null && outputScores.length > 0) {
          int len = Math.min(fullCount, outputScores[0].length);
          LOGGER.i("FULL_DUMP scores[0.." + len + "]=" + Arrays.toString(Arrays.copyOfRange(outputScores[0], 0, len)));
        }
        if (outputClasses != null && outputClasses.length > 0) {
          int len = Math.min(fullCount, outputClasses[0].length);
          LOGGER.i("FULL_DUMP classes[0.." + len + "]=" + Arrays.toString(Arrays.copyOfRange(outputClasses[0], 0, len)));
        }
        if (outputLocations != null && outputLocations.length > 0) {
          int boxCount = Math.min(fullCount, outputLocations[0].length);
          for (int b = 0; b < boxCount; ++b) {
            LOGGER.i("FULL_DUMP box[" + b + "]=" + Arrays.toString(outputLocations[0][b]));
          }
        }
      } catch (Exception ex) {
        LOGGER.i("FULL_DUMP error: " + ex.getMessage());
      }
    }

    // If the model produced raw logits per-anchor (e.g. [1, anchors, classes]),
    // compute per-anchor scores (softmax) and classes, then perform top-k + NMS
    // to produce final detections.
    int[] selectedIndices = null;
    if (outputLogits != null) {
      // Dump a small sample of raw logits to help debug unexpected uniform scores.
      if (FULL_DUMP_OUTPUTS) {
        try {
          int sampleAnchors = Math.min(5, outputLogits[0].length);
          StringBuilder sb = new StringBuilder();
          for (int a = 0; a < sampleAnchors; ++a) {
            sb.append("anchor").append(a).append(":");
            for (int c = 0; c < outputLogits[0][a].length; ++c) {
              sb.append(outputLogits[0][a][c]).append(",");
            }
            sb.append(";");
          }
          LOGGER.i("RAW_LOGITS sample=" + sb.toString());
          try {
            if (idxLogits != -1) {
              DataType dt = tfLite.getOutputTensor(idxLogits).dataType();
              LOGGER.i("RAW_LOGITS dtype=" + dt);
            }
          } catch (Exception ignored) {}
        } catch (Exception ex) {
          LOGGER.i("RAW_LOGITS dump error: " + ex.getMessage());
        }
      }
      try {
        final int anchors = outputLogits[0].length;
        final int numClasses = outputLogits[0][0].length;
        // allocate scores/classes per-anchor
        outputScores = new float[1][anchors];
        outputClasses = new float[1][anchors];

        // For each anchor, compute softmax over classes and pick best class & score.
        for (int a = 0; a < anchors; ++a) {
          // compute softmax for this anchor
          float max = Float.NEGATIVE_INFINITY;
          for (int c = 0; c < numClasses; ++c) {
            float v = outputLogits[0][a][c];
            if (v > max) max = v;
          }
          double sum = 0.0;
          double[] exps = new double[numClasses];
          for (int c = 0; c < numClasses; ++c) {
            double e = Math.exp(outputLogits[0][a][c] - max);
            exps[c] = e;
            sum += e;
          }
          int bestC = 0;
          double bestP = -1.0;
          for (int c = 0; c < numClasses; ++c) {
            double p = exps[c] / sum;
            if (p > bestP) { bestP = p; bestC = c; }
          }
          outputScores[0][a] = (float) bestP;
          outputClasses[0][a] = (float) bestC;
        }

        // Now select top candidates by score and apply NMS.
        // Build index array
        Integer[] idxs = new Integer[anchors];
        for (int i = 0; i < anchors; ++i) idxs[i] = i;
        java.util.Arrays.sort(idxs, (i1, i2) -> Float.compare(outputScores[0][i2], outputScores[0][i1]));

        final float scoreThreshold = 0.01f; // allow low threshold for debugging
        final float iouThreshold = 0.5f;
        boolean[] removed = new boolean[anchors];
        java.util.ArrayList<Integer> picks = new java.util.ArrayList<>();
        for (int id : idxs) {
          if (outputScores[0][id] < scoreThreshold) break; // remaining scores are too low
          if (removed[id]) continue;
          // pick this box
          picks.add(id);
          // suppress overlaps
          for (int j = 0; j < anchors; ++j) {
            if (removed[j]) continue;
            if (j == id) continue;
            // compute IoU between box id and j
            float[] boxA = outputLocations[0][id];
            float[] boxB = outputLocations[0][j];
            float iou = computeIoU(boxA, boxB);
            if (iou > iouThreshold) removed[j] = true;
          }
          if (picks.size() >= NUM_DETECTIONS) break;
        }

        selectedIndices = new int[picks.size()];
        for (int i = 0; i < picks.size(); ++i) selectedIndices[i] = picks.get(i);
        // Update reported numDetections for downstream code/diagnostics
        if (numDetections == null || numDetections.length == 0) numDetections = new float[1];
        numDetections[0] = selectedIndices.length;
        // Log postprocessing summary so we can verify selected detections and scores.
        try {
          LOGGER.i("POSTPROC selectedCount=" + selectedIndices.length);
          int toShow = Math.min(selectedIndices.length, 5);
          for (int k = 0; k < toShow; ++k) {
            int id = selectedIndices[k];
            String lbl = resolveLabel((int) outputClasses[0][id]);
            LOGGER.i("POSTPROC[" + k + "] idx=" + id + " label=" + lbl + " score=" + outputScores[0][id] + " box=" + Arrays.toString(outputLocations[0][id]));
          }
        } catch (Exception ignored) {}
      } catch (Exception ex) {
        LOGGER.i("Postprocess logits error: " + ex.getMessage());
      }
    }

    // Show the best detections.
    // after scaling them back to the input size.
      
    // You need to use the number of detections from the output and not the NUM_DETECTONS variable declared on top
      // because on some models, they don't always output the same total number of detections
      // For example, your model's NUM_DETECTIONS = 20, but sometimes it only outputs 16 predictions
      // If you don't use the output's numDetections, you'll get nonsensical data
    // Decide which indices to use for final detections. If we performed
    // SSD-style postprocessing we have selectedIndices filled; otherwise
    // we fall back to model's reported numDetections and first N entries.
    final ArrayList<Recognition> recognitions = new ArrayList<>();
    if (selectedIndices != null && selectedIndices.length > 0) {
      int count = Math.min(NUM_DETECTIONS, selectedIndices.length);
      for (int k = 0; k < count; ++k) {
        int i = selectedIndices[k];
        final RectF detection =
            new RectF(
                outputLocations[0][i][1] * inputSize,
                outputLocations[0][i][0] * inputSize,
                outputLocations[0][i][3] * inputSize,
                outputLocations[0][i][2] * inputSize);
        int classIndex = (int) outputClasses[0][i];
        String label = resolveLabel(classIndex);
        recognitions.add(new Recognition("" + i, label, outputScores[0][i], detection));
      }
    } else {
      int numDetectionsOutput = Math.min(NUM_DETECTIONS, (int) numDetections[0]); // cast from float to integer, use min for safety
      for (int i = 0; i < numDetectionsOutput; ++i) {
        final RectF detection =
            new RectF(
                outputLocations[0][i][1] * inputSize,
                outputLocations[0][i][0] * inputSize,
                outputLocations[0][i][3] * inputSize,
                outputLocations[0][i][2] * inputSize);
        int classIndex = (int) outputClasses[0][i];
        String label = resolveLabel(classIndex);
        recognitions.add(new Recognition("" + i, label, outputScores[0][i], detection));
      }
    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  // Resolve label robustly (try 1-based then 0-based then fallback)
  private String resolveLabel(int classIndex) {
    int labelIndex = classIndex - 1; // many TF models output 1-based class ids
    if (labelIndex >= 0 && labelIndex < labels.size()) {
      return labels.get(labelIndex);
    } else if (classIndex >= 0 && classIndex < labels.size()) {
      return labels.get(classIndex);
    } else if (!labels.isEmpty()) {
      return labels.get(0);
    }
    return "unknown";
  }

  // Compute IoU for boxes in [ymin, xmin, ymax, xmax] normalized coordinates
  private static float computeIoU(final float[] a, final float[] b) {
    float ay1 = a[0], ax1 = a[1], ay2 = a[2], ax2 = a[3];
    float by1 = b[0], bx1 = b[1], by2 = b[2], bx2 = b[3];
    float interY1 = Math.max(ay1, by1);
    float interX1 = Math.max(ax1, bx1);
    float interY2 = Math.min(ay2, by2);
    float interX2 = Math.min(ax2, bx2);
    float interH = interY2 - interY1;
    float interW = interX2 - interX1;
    if (interH <= 0 || interW <= 0) return 0f;
    float interArea = interH * interW;
    float areaA = Math.max(0f, ay2 - ay1) * Math.max(0f, ax2 - ax1);
    float areaB = Math.max(0f, by2 - by1) * Math.max(0f, bx2 - bx1);
    float union = areaA + areaB - interArea;
    if (union <= 0f) return 0f;
    return interArea / union;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {}

  public void setNumThreads(int num_threads) {
    // Recreate the interpreter with the requested thread count and the current processor type.
    if (modelFileBuffer == null) {
      LOGGER.i("setNumThreads: model file buffer is null, cannot recreate interpreter");
      return;
    }
    try {
      Interpreter.Options options = new Interpreter.Options();
      // Re-apply delegate selection based on currentProcessorType
      switch (currentProcessorType) {
        case NPU:
          try {
            NnApiDelegate nnApiDelegate = new NnApiDelegate();
            options.addDelegate(nnApiDelegate);
          } catch (Exception e) {
            LOGGER.i("Failed to add NNAPI delegate: " + e.getMessage());
          }
          break;
        case GPU:
          try {
            GpuDelegate gpuDelegate = new GpuDelegate();
            options.addDelegate(gpuDelegate);
          } catch (Exception e) {
            LOGGER.i("Failed to add GPU delegate: " + e.getMessage());
          }
          break;
        case CPU:
        default:
          // no delegate
          break;
      }
      try { options.setNumThreads(num_threads); } catch (Exception ignored) {}
      if (tfLite != null) {
        try { tfLite.close(); } catch (Exception ignored) {}
      }
      tfLite = new Interpreter(modelFileBuffer, options);
    } catch (Exception e) {
      LOGGER.i("Failed to recreate interpreter with new thread count: " + e.getMessage());
    }
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    // Toggle NNAPI by recreating the interpreter with NNAPI delegate when requested.
    if (modelFileBuffer == null) {
      LOGGER.i("setUseNNAPI: model file buffer is null, cannot recreate interpreter");
      return;
    }
    try {
      Interpreter.Options options = new Interpreter.Options();
      if (isChecked) {
        try {
          NnApiDelegate nnApiDelegate = new NnApiDelegate();
          options.addDelegate(nnApiDelegate);
          currentProcessorType = ProcessorType.NPU;
        } catch (Exception e) {
          LOGGER.i("Failed to create NNAPI delegate: " + e.getMessage());
        }
      } else {
        currentProcessorType = ProcessorType.CPU;
      }
      try { options.setNumThreads(NUM_THREADS); } catch (Exception ignored) {}
      if (tfLite != null) {
        try { tfLite.close(); } catch (Exception ignored) {}
      }
      tfLite = new Interpreter(modelFileBuffer, options);
    } catch (Exception e) {
      LOGGER.i("Failed to recreate interpreter for NNAPI change: " + e.getMessage());
    }
  }
}
