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

package org.tensorflow.lite.examples.detection.tracking;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.text.TextUtils;
import android.util.Pair;
import android.util.TypedValue;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier.Recognition;

/** A tracker that handles non-max suppression and matches existing objects to new detections. */
public class MultiBoxTracker {
  private static final float TEXT_SIZE_DIP = 18;
  private static final float MIN_SIZE = 16.0f;
  private static final int[] COLORS = {
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.CYAN,
    Color.MAGENTA,
    Color.WHITE,
    Color.parseColor("#55FF55"),
    Color.parseColor("#FFA500"),
    Color.parseColor("#FF8888"),
    Color.parseColor("#AAAAFF"),
    Color.parseColor("#FFFFAA"),
    Color.parseColor("#55AAAA"),
    Color.parseColor("#AA33AA"),
    Color.parseColor("#0D0068")
  };
  final List<Pair<Float, RectF>> screenRects = new LinkedList<Pair<Float, RectF>>();
  private final Logger logger = new Logger();
  private final Queue<Integer> availableColors = new LinkedList<Integer>();
  private final List<TrackedRecognition> trackedObjects = new LinkedList<TrackedRecognition>();
  private final Paint boxPaint = new Paint();
  private final float textSizePx;
  private final BorderedText borderedText;
  private Matrix frameToCanvasMatrix;
  private int frameWidth;
  private int frameHeight;
  private int sensorOrientation;
  private float zoomFactor = 1.5f;

  public MultiBoxTracker(final Context context) {
    for (final int color : COLORS) {
      availableColors.add(color);
    }

    boxPaint.setColor(Color.RED);
    boxPaint.setStyle(Style.STROKE);
    boxPaint.setStrokeWidth(3.0f);
    boxPaint.setStrokeCap(Cap.ROUND);
    boxPaint.setStrokeJoin(Join.ROUND);
    boxPaint.setStrokeMiter(100);

    textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
  }

  public synchronized void setFrameConfiguration(
      final int width, final int height, final int sensorOrientation) {
    frameWidth = width;
    frameHeight = height;
    this.sensorOrientation = sensorOrientation;
  }
  
  public synchronized void setZoomFactor(float zoom) {
    this.zoomFactor = zoom;
  }

  public synchronized void drawDebug(final Canvas canvas) {
    final Paint textPaint = new Paint();
    textPaint.setColor(Color.WHITE);
    textPaint.setTextSize(60.0f);

    final Paint boxPaint = new Paint();
    boxPaint.setColor(Color.RED);
    boxPaint.setAlpha(200);
    boxPaint.setStyle(Style.STROKE);
    boxPaint.setStrokeWidth(8.0f);    // viền của bounding box

    for (final Pair<Float, RectF> detection : screenRects) {
      final RectF rect = new RectF(detection.second);
      
      // Mở rộng bounding box thêm 20px về 4 góc
//       rect.left -= 8;
//       rect.top -= 10;
//       rect.right += 8;
//       rect.bottom += 10;
      
      canvas.drawRect(rect, boxPaint);
      canvas.drawText("" + detection.first, rect.left, rect.top, textPaint);
      borderedText.drawText(canvas, rect.centerX(), rect.centerY(), "" + detection.first);
    }
  }

  public synchronized void trackResults(final List<Recognition> results, final long timestamp) {
    logger.i("Processing %d results from %d", results.size(), timestamp);
    processResults(results);
  }

  private Matrix getFrameToCanvasMatrix() {
    return frameToCanvasMatrix;
  }

  public synchronized void draw(final Canvas canvas) {
    final boolean rotated = sensorOrientation % 180 == 90;
    
    // Use Math.max() for FILL mode - same as canvas scaling
    final float multiplier =
        Math.max(
            canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
            canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
    
    // Debug log - Resolution info
    logger.d("=== ZOOM DEBUG INFO ===");
    logger.d("Original frame size: %dx%d", frameWidth, frameHeight);
    logger.d("Canvas size: %dx%d", canvas.getWidth(), canvas.getHeight());
    logger.d("Rotated: %s", rotated);
    logger.d("Base multiplier: %.3f (FILL MODE - Math.max)", multiplier);
    
    // SIMPLIFIED: No extra zoom factor, use canvas size directly for perfect alignment
    int finalWidth = canvas.getWidth();   // Match canvas exactly
    int finalHeight = canvas.getHeight(); // Match canvas exactly
    logger.d("Final transformed size: %dx%d (CANVAS MATCH)", finalWidth, finalHeight);
    
    // No offset needed since we match canvas size exactly
    logger.d("Zoom offset: NONE (perfect canvas match)");
    
    // Create transformation matrix that matches canvas scaling exactly
    frameToCanvasMatrix =
        ImageUtils.getTransformationMatrix(
            frameWidth,
            frameHeight,
            finalWidth,  // Use canvas width
            finalHeight, // Use canvas height  
            sensorOrientation,
            false);
            
    for (final TrackedRecognition recognition : trackedObjects) {
      final RectF trackedPos = new RectF(recognition.location);
      
      // Debug log - Box coordinates before transformation
      logger.d("Box BEFORE transform: [%.1f, %.1f, %.1f, %.1f] %s", 
               recognition.location.left, recognition.location.top, 
               recognition.location.right, recognition.location.bottom,
               recognition.title);

      // Apply the transformation matrix that matches canvas scaling exactly
      getFrameToCanvasMatrix().mapRect(trackedPos);
      
      // RỘNG BOUNDING BOX LÊN PHÍA TRÊN - giảm giá trị top để mở rộng lên
      trackedPos.top -= 0.03f * Math.abs(trackedPos.top);
      
      // Debug log - Box coordinates after transformation (FINAL - no more adjustments)
      logger.d("Box AFTER transform (FINAL): [%.1f, %.1f, %.1f, %.1f]", 
               trackedPos.left, trackedPos.top, 
               trackedPos.right, trackedPos.bottom);
      
      // NO MORE ADJUSTMENTS - box should be perfectly aligned with canvas now
      
      boxPaint.setColor(recognition.color);

      float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;
      canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

      final String labelString =
          !TextUtils.isEmpty(recognition.title)
              ? String.format("%s %.2f", recognition.title, (100 * recognition.detectionConfidence))
              : String.format("%.2f", (100 * recognition.detectionConfidence));
      //            borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.top,
      // labelString);
      borderedText.drawText(
          canvas, trackedPos.left + cornerSize, trackedPos.top, labelString + "%", boxPaint);
    }
    logger.d("======================");
  }

  private void processResults(final List<Recognition> results) {
    final List<Pair<Float, Recognition>> rectsToTrack = new LinkedList<Pair<Float, Recognition>>();

    screenRects.clear();
    final Matrix rgbFrameToScreen = new Matrix(getFrameToCanvasMatrix());

    for (final Recognition result : results) {
      if (result.getLocation() == null) {
        continue;
      }
      final RectF detectionFrameRect = new RectF(result.getLocation());

      final RectF detectionScreenRect = new RectF();
      rgbFrameToScreen.mapRect(detectionScreenRect, detectionFrameRect);

      logger.v(
          "Result! Frame: " + result.getLocation() + " mapped to screen:" + detectionScreenRect);

      screenRects.add(new Pair<Float, RectF>(result.getConfidence(), detectionScreenRect));

      if (detectionFrameRect.width() < MIN_SIZE || detectionFrameRect.height() < MIN_SIZE) {
        logger.w("Degenerate rectangle! " + detectionFrameRect);
        continue;
      }

      rectsToTrack.add(new Pair<Float, Recognition>(result.getConfidence(), result));
    }

    trackedObjects.clear();
    if (rectsToTrack.isEmpty()) {
      logger.v("Nothing to track, aborting.");
      return;
    }

    // xóa vòng for mếu như k cần lọc nested box
    
    List<Pair<Float, Recognition>> filteredRects = new LinkedList<>();
    
    for (int i = 0; i < rectsToTrack.size(); i++) {
      Pair<Float, Recognition> current = rectsToTrack.get(i);
      RectF currentRect = current.second.getLocation();
      boolean isNested = false;
      
      // Check if current box is completely inside any other box
      for (int j = 0; j < rectsToTrack.size(); j++) {
        if (i == j) continue; // Skip self comparison
        
        Pair<Float, Recognition> other = rectsToTrack.get(j);
        RectF otherRect = other.second.getLocation();
        
        // Check if current box is completely contained within other box
        if (isCompletelyInside(currentRect, otherRect)) {
          logger.d("Box %s [%.1f,%.1f,%.1f,%.1f] is nested inside %s [%.1f,%.1f,%.1f,%.1f] - removing nested box",
                   current.second.getTitle(), currentRect.left, currentRect.top, currentRect.right, currentRect.bottom,
                   other.second.getTitle(), otherRect.left, otherRect.top, otherRect.right, otherRect.bottom);
          isNested = true;
          break;
        }
      }
      
      if (!isNested) {
        filteredRects.add(current);
      }
    }
    
    logger.d("Filtered boxes: %d -> %d (removed %d nested boxes)", 
             rectsToTrack.size(), filteredRects.size(), rectsToTrack.size() - filteredRects.size());


    for (final Pair<Float, Recognition> potential : filteredRects) {
      final TrackedRecognition trackedRecognition = new TrackedRecognition();
      trackedRecognition.detectionConfidence = potential.first;
      trackedRecognition.location = new RectF(potential.second.getLocation());
      trackedRecognition.title = potential.second.getTitle();
      trackedRecognition.color = COLORS[trackedObjects.size()];
      trackedObjects.add(trackedRecognition);

      if (trackedObjects.size() >= COLORS.length) {
        break;
      }
    }
  }

  // Helper method to check if one rectangle is completely inside another
  private boolean isCompletelyInside(RectF inner, RectF outer) {
    return inner.left >= outer.left && 
           inner.top >= outer.top && 
           inner.right <= outer.right && 
           inner.bottom <= outer.bottom;
  }

  private static class TrackedRecognition {
    RectF location;
    float detectionConfidence;
    int color;
    String title;
  }
}
