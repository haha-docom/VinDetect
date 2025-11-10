/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.Manifest;
import android.app.Fragment;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.pm.ActivityInfo;
import android.content.res.Configuration;
import android.graphics.Color;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.Trace;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.SwitchCompat;
import androidx.appcompat.widget.Toolbar;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.view.WindowInsetsControllerCompat;

import android.util.Size;
import android.view.Surface;
import android.hardware.usb.UsbDevice;
import com.jiangdg.ausbc.MultiCameraClient;
import com.jiangdg.ausbc.callback.IDeviceConnectCallBack;
import com.jiangdg.usb.USBMonitor.UsbControlBlock;
import com.jiangdg.uvc.UVCCamera;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.CompoundButton;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.Spinner;
import android.widget.AdapterView;
import android.widget.FrameLayout;
import android.widget.Button;
import com.google.android.material.bottomsheet.BottomSheetBehavior;
import java.nio.ByteBuffer;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.ProcessorType;

public abstract class CameraActivity extends AppCompatActivity
    implements OnImageAvailableListener,
        Camera.PreviewCallback,
        CompoundButton.OnCheckedChangeListener,
        View.OnClickListener {
  private static final Logger LOGGER = new Logger();

  private static final int PERMISSIONS_REQUEST = 1;

  private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
  protected int previewWidth = 0;
  protected int previewHeight = 0;
  private boolean debug = false;
  private Handler handler;
  private HandlerThread handlerThread;
  private boolean useCamera2API;
  private boolean isProcessingFrame = false;
  private byte[][] yuvBytes = new byte[3][];
  private int[] rgbBytes = null;
  private int yRowStride;
  private Runnable postInferenceCallback;
  private Runnable imageConverter;

  // USB / UVC fields
  private MultiCameraClient mCameraClient;
  private UVCCamera mUVCCamera;

  private LinearLayout bottomSheetLayout;
  private LinearLayout gestureLayout;
  private BottomSheetBehavior<LinearLayout> sheetBehavior;

  // Right sheet cho landscape mode
  private LinearLayout rightSheetLayout;
  private LinearLayout rightGestureLayout;
  protected TextView frameValueTextViewRight, cropValueTextViewRight, inferenceTimeTextViewRight;
  protected ImageView rightSheetArrowImageView;
  private ImageView plusImageViewRight, minusImageViewRight;
  private ImageView zoomInImageViewRight, zoomOutImageViewRight;
  private TextView zoomValueTextViewRight;
  private SwitchCompat apiSwitchCompatRight;
  private ImageButton rotateScreenButtonRight;
  private TextView threadsTextViewRight;
  private Spinner processorSpinnerRight;

  // Right sheet state tracking
  private boolean isRightSheetExpanded = true;
  private boolean isRightSheetHidden = false;
  private int rightSheetCollapsedWidth = 60; // Width when collapsed (only gesture area visible)

  // Zoom variables
  private int currentZoomLevel = 100; // Default zoom 100%
  private static final int MIN_ZOOM = 100;
  private static final int MAX_ZOOM = 300;
  private static final int ZOOM_STEP = 10;

  protected TextView frameValueTextView, cropValueTextView, inferenceTimeTextView;
  protected ImageView bottomSheetArrowImageView;
  private ImageView plusImageView, minusImageView;
  private SwitchCompat apiSwitchCompat;
  private Button rotateScreenButton;
  private TextView threadsTextView;
  private Spinner processorSpinner;
  
  // Confidence threshold controls
  private SeekBar confidenceSeekBar;
  private TextView confidenceValueTextView;
  private SeekBar confidenceSeekBarRight;
  private TextView confidenceValueTextViewRight;

  @Override
  protected void onCreate(final Bundle savedInstanceState) {
    LOGGER.d("onCreate " + this);
    super.onCreate(null);
    
    // FORCE full screen layout - remove all system UI space completely
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS);
    
    // Additional flags inspired by TabSpec approach to remove bottom lines/spaces
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN);
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_LAYOUT_INSET_DECOR);
    
    // Ẩn navigation bar (3 nút) ngay khi khởi động app
    hideSystemUI();
    WindowCompat.setDecorFitsSystemWindows(getWindow(), false);

    WindowInsetsControllerCompat controller =
        new WindowInsetsControllerCompat(getWindow(), getWindow().getDecorView());

    // Ẩn system bars
    controller.hide(WindowInsetsCompat.Type.systemBars());

    // Cho phép vuốt để hiện tạm thời
    controller.setSystemBarsBehavior(
        WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
    );

    // Đè màu trong suốt (tránh black line)
    getWindow().setStatusBarColor(Color.TRANSPARENT);
    getWindow().setNavigationBarColor(Color.TRANSPARENT);

    
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

    setContentView(R.layout.tfe_od_activity_camera);
//    Toolbar toolbar = findViewById(R.id.toolbar);
//    setSupportActionBar(toolbar);
//    getSupportActionBar().setDisplayShowTitleEnabled(false);

    if (hasPermission()) {
      setFragment();
    } else {
      requestPermission();
    }

    threadsTextView = findViewById(R.id.threads);
    plusImageView = findViewById(R.id.plus);
    minusImageView = findViewById(R.id.minus);
    apiSwitchCompat = findViewById(R.id.api_info_switch);
    processorSpinner = findViewById(R.id.processor_spinner);
    
    // Khởi tạo UI dựa trên orientation
    initializeUI();

    // Lấy reference tới container camera
    FrameLayout container = findViewById(R.id.container);
    
    // Thêm touch listener cho container để hiển thị lại navigation bar và sheet khi bị ẩn
    container.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        // Hiển thị lại navigation bar và status bar khi chạm vào màn hình
        showSystemUI();
        
        int orientation = getResources().getConfiguration().orientation;
        if (orientation == android.content.res.Configuration.ORIENTATION_LANDSCAPE) {
          // Trong landscape mode, hiện lại right sheet nếu bị ẩn
          if (isRightSheetHidden) {
            showRightSheet();
          }
        } else {
          // Trong portrait mode, hiện lại bottom sheet nếu bị ẩn
          if (sheetBehavior != null && sheetBehavior.getState() == BottomSheetBehavior.STATE_HIDDEN) {
            sheetBehavior.setState(BottomSheetBehavior.STATE_COLLAPSED);
          }
        }
      }
    });

    frameValueTextView = findViewById(R.id.frame_info);
    cropValueTextView = findViewById(R.id.crop_info);
    inferenceTimeTextView = findViewById(R.id.inference_info);

    if (apiSwitchCompat != null) {
      apiSwitchCompat.setOnCheckedChangeListener(this);
    }

    if (plusImageView != null) {
      plusImageView.setOnClickListener(this);
    }
    if (minusImageView != null) {
      minusImageView.setOnClickListener(this);
    }
    if (rotateScreenButton != null) {
      rotateScreenButton.setOnClickListener(this);
    }
    
    // Setup processor spinner
    if (processorSpinner != null) {
      processorSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
        @Override
        public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
          onProcessorChanged(ProcessorType.fromValue(position));
        }

        @Override
        public void onNothingSelected(AdapterView<?> parent) {
        }
      });
    }
    // Initialize MultiCameraClient (libausbc) to detect and handle USB UVC cameras.
    mCameraClient =
        new MultiCameraClient(
            this,
            new IDeviceConnectCallBack() {
              @Override
              public void onAttachDev(UsbDevice device) {
                LOGGER.i("USB device attached: %s", device == null ? "null" : device.getDeviceName());
                // Auto-request permission could be done here if desired:
                try {
                  mCameraClient.requestPermission(device);
                } catch (final Exception e) {
                  LOGGER.e(e, "requestPermission failed");
                }
              }

              @Override
              public void onDetachDec(UsbDevice device) {
                LOGGER.i("USB device detached: %s", device == null ? "null" : device.getDeviceName());
              }

              @Override
              public void onConnectDev(UsbDevice device, UsbControlBlock ctrlBlock) {
                LOGGER.i("USB device connected: %s", device == null ? "null" : device.getDeviceName());
                 try {
    if (ctrlBlock != null) {
      LOGGER.i(
          "UsbControlBlock: fd=%d, bus=%d, dev=%d, vendorId=%d, productId=%d, name=%s",
          ctrlBlock.getFileDescriptor(),
          ctrlBlock.getBusNum(),
          ctrlBlock.getDevNum(),
          ctrlBlock.getVenderId(),
          ctrlBlock.getProductId(),
          ctrlBlock.getDeviceName());
    } else {
      LOGGER.w("UsbControlBlock is null");
    }
  } catch (Exception e) {
    LOGGER.e(e, "Failed to log UsbControlBlock details");
  }
                try {
                  if (mUVCCamera == null) {
                    mUVCCamera = new UVCCamera();
                  }
                  mUVCCamera.open(ctrlBlock);

                  // Configure preview size and notify the activity so UI/layout can adapt.
                  try {
                    // Set fixed 640x480 for consistent performance and proper aspect ratio
                    previewWidth = 640;
                    previewHeight = 480;
                    rgbBytes = new int[previewWidth * previewHeight];
                    
                    // Set UVC camera to use 640x480
                    mUVCCamera.setPreviewSize(previewWidth, previewHeight);
                    LOGGER.i("Set UVC camera preview size to %dx%d", previewWidth, previewHeight);
                    
                    // notify subclass about preview size on UI thread
                    runOnUiThread(new Runnable() {
                      @Override
                      public void run() {
                        CameraActivity.this.onPreviewSizeChosen(new Size(previewWidth, previewHeight), getScreenOrientation());
                        
                        // Áp dụng auto zoom cho chế độ landscape để loại bỏ khoảng đen
                        applyLandscapeAutoZoom();
                      }
                    });
                  } catch (final Exception e) {
                    LOGGER.w(e, "Failed to set preview size");
                  }

                                    // Register frame callback to receive raw NV21 frames and feed into existing pipeline
                  mUVCCamera.setFrameCallback(
                      new com.jiangdg.uvc.IFrameCallback() {
                        @Override
                        public void onFrame(final ByteBuffer frame) {
                          if (isProcessingFrame) {
                            LOGGER.w("Dropping frame!");
                            return;
                          }

                          try {
                            isProcessingFrame = true;

                            // ensure rgb buffer exists
                            if (rgbBytes == null && previewWidth > 0 && previewHeight > 0) {
                              rgbBytes = new int[previewWidth * previewHeight];
                            }

                            final int capacity = frame.remaining();
                            if (yuvBytes[0] == null || yuvBytes[0].length != capacity) {
                              yuvBytes[0] = new byte[capacity];
                            }
                            frame.get(yuvBytes[0]);

                            // row stride for NV21 should be image width
                            yRowStride = previewWidth;

                            imageConverter =
                                new Runnable() {
                                  @Override
                                  public void run() {
                                    ImageUtils.convertYUV420SPToARGB8888(yuvBytes[0], previewWidth, previewHeight, rgbBytes);
                                  }
                                };

                            postInferenceCallback =
                                new Runnable() {
                                  @Override
                                  public void run() {
                                    isProcessingFrame = false;
                                  }
                                };

                            // hand off to existing processing flow
                            processImage();
                          } catch (final Exception e) {
                            LOGGER.e(e, "Exception in frame callback");
                            isProcessingFrame = false;
                          }
                        }
                      },
                      UVCCamera.PIXEL_FORMAT_NV21);

                  // Attach preview to the TextureView so user sees camera feed on screen
                  runOnUiThread(
                      new Runnable() {
                        @Override
                        public void run() {
                          try {
                            android.view.View v = findViewById(R.id.texture);
                            if (v instanceof android.view.TextureView) {
                              final android.view.TextureView tv = (android.view.TextureView) v;
                              android.graphics.SurfaceTexture st = tv.getSurfaceTexture();
                              if (st == null) {
                                // TextureView not ready yet, wait for it
                                tv.setSurfaceTextureListener(
                                    new android.view.TextureView.SurfaceTextureListener() {
                                      @Override
                                      public void onSurfaceTextureAvailable(
                                          android.graphics.SurfaceTexture surface, int width, int height) {
                                        try {
                                          mUVCCamera.setPreviewTexture(surface);
                                          mUVCCamera.startPreview();
                                          LOGGER.i("UVC preview started via listener");
                                        } catch (final Exception e) {
                                          LOGGER.e(e, "Failed to setPreviewTexture/startPreview (listener)");
                                        }
                                      }

                                      @Override
                                      public void onSurfaceTextureSizeChanged(
                                          android.graphics.SurfaceTexture surface, int width, int height) {}

                                      @Override
                                      public boolean onSurfaceTextureDestroyed(
                                          android.graphics.SurfaceTexture surface) {
                                        try {
                                          if (mUVCCamera != null) mUVCCamera.stopPreview();
                                        } catch (Exception ignored) {}
                                        return true;
                                      }

                                      @Override
                                      public void onSurfaceTextureUpdated(
                                          android.graphics.SurfaceTexture surface) {}
                                    });
                              } else {
                                // TextureView ready, start preview immediately
                                try {
                                  // Scale TextureView để fill toàn màn hình và loại bỏ viền đen
                              
                                  mUVCCamera.setPreviewTexture(st);
                                  mUVCCamera.startPreview();
                                  LOGGER.i("UVC preview started immediately");
                                } catch (final Exception e) {
                                  LOGGER.e(e, "Failed to setPreviewTexture/startPreview");
                                }
                              }
                            } else {
                              // Fallback: off-screen preview for frame callback only
                              try {
                                android.graphics.SurfaceTexture off = new android.graphics.SurfaceTexture(0);
                                mUVCCamera.setPreviewTexture(off);
                                mUVCCamera.startPreview();
                                LOGGER.w("Using off-screen preview (no display)");
                              } catch (final Exception e) {
                                LOGGER.e(e, "Failed to startPreview on fallback surface");
                              }
                            }
                          } catch (final Exception e) {
                            LOGGER.e(e, "Preview attach failed");
                          }
                        }
                      });
                } catch (final Exception e) {
                  LOGGER.e(e, "Failed to open UVCCamera");
                }
              }

              @Override
              public void onDisConnectDec(UsbDevice device, UsbControlBlock ctrlBlock) {
                LOGGER.i("USB device disconnected: %s", device == null ? "null" : device.getDeviceName());
                if (mUVCCamera != null) {
                  try {
                    mUVCCamera.close();
                  } catch (final Exception e) {
                    LOGGER.e(e, "Failed to close UVCCamera");
                  }
                  mUVCCamera = null;
                }
              }

              @Override
              public void onCancelDev(UsbDevice device) {
                LOGGER.i("USB device permission canceled: %s", device == null ? "null" : device.getDeviceName());
              }
            });
    // start listening for device attach/detach and permissions
    try {
      mCameraClient.register();
    } catch (final Exception e) {
      LOGGER.e(e, "Failed to register MultiCameraClient");
    }
  }

  protected int[] getRgbBytes() {
    imageConverter.run();
    return rgbBytes;
  }

  protected int getLuminanceStride() {
    return yRowStride;
  }

  protected byte[] getLuminance() {
    return yuvBytes[0];
  }

  /** Callback for android.hardware.Camera API */
  @Override
  public void onPreviewFrame(final byte[] bytes, final Camera camera) {
    if (isProcessingFrame) {
      LOGGER.w("Dropping frame!");
      return;
    }

    try {
      // Initialize the storage bitmaps once when the resolution is known.
      if (rgbBytes == null) {
        Camera.Size previewSize = camera.getParameters().getPreviewSize();
        previewHeight = previewSize.height;
        previewWidth = previewSize.width;
        rgbBytes = new int[previewWidth * previewHeight];
        onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
      }
    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
      return;
    }

    isProcessingFrame = true;
    yuvBytes[0] = bytes;
    yRowStride = previewWidth;

    imageConverter =
        new Runnable() {
          @Override
          public void run() {
            ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes);
          }
        };

    postInferenceCallback =
        new Runnable() {
          @Override
          public void run() {
            camera.addCallbackBuffer(bytes);
            isProcessingFrame = false;
          }
        };
    processImage();
  }

  /** Callback for Camera2 API */
  @Override
  public void onImageAvailable(final ImageReader reader) {
    // We need wait until we have some size from onPreviewSizeChosen
    if (previewWidth == 0 || previewHeight == 0) {
      return;
    }
    if (rgbBytes == null) {
      rgbBytes = new int[previewWidth * previewHeight];
    }
    try {
      final Image image = reader.acquireLatestImage();

      if (image == null) {
        return;
      }

      if (isProcessingFrame) {
        image.close();
        return;
      }
      isProcessingFrame = true;
      Trace.beginSection("imageAvailable");
      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);
      yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();

      imageConverter =
          new Runnable() {
            @Override
            public void run() {
              ImageUtils.convertYUV420ToARGB8888(
                  yuvBytes[0],
                  yuvBytes[1],
                  yuvBytes[2],
                  previewWidth,
                  previewHeight,
                  yRowStride,
                  uvRowStride,
                  uvPixelStride,
                  rgbBytes);
            }
          };

      postInferenceCallback =
          new Runnable() {
            @Override
            public void run() {
              image.close();
              isProcessingFrame = false;
            }
          };

      processImage();
    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }
    Trace.endSection();
  }

  @Override
  public synchronized void onStart() {
    LOGGER.d("onStart " + this);
    super.onStart();
  }

  @Override
  public synchronized void onResume() {
    LOGGER.d("onResume " + this);
    super.onResume();

    handlerThread = new HandlerThread("inference");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
    
    // Apply landscape auto zoom when resuming the activity
    new Handler(getMainLooper()).postDelayed(new Runnable() {
      @Override
      public void run() {
        applyLandscapeAutoZoom();

      }
    }, 1000); // 1 second delay to ensure camera is ready
  }

  @Override
  public synchronized void onPause() {
    LOGGER.d("onPause " + this);

    handlerThread.quitSafely();
    try {
      handlerThread.join();
      handlerThread = null;
      handler = null;
    } catch (final InterruptedException e) {
      LOGGER.e(e, "Exception!");
    }

    super.onPause();
  }

  @Override
  public synchronized void onStop() {
    LOGGER.d("onStop " + this);
    super.onStop();
  }

  @Override
  public synchronized void onDestroy() {
    LOGGER.d("onDestroy " + this);
    // cleanup USB / UVC resources
    if (mCameraClient != null) {
      try {
        mCameraClient.unRegister();
      } catch (final Exception e) {
        LOGGER.e(e, "Failed to unRegister MultiCameraClient");
      }
      try {
        mCameraClient.destroy();
      } catch (final Exception e) {
        LOGGER.e(e, "Failed to destroy MultiCameraClient");
      }
      mCameraClient = null;
    }
    if (mUVCCamera != null) {
      try {
        mUVCCamera.destroy();
      } catch (final Exception e) {
        LOGGER.e(e, "Failed to destroy UVCCamera");
      }
      mUVCCamera = null;
    }

    super.onDestroy();
  }

  protected synchronized void runInBackground(final Runnable r) {
    if (handler != null) {
      handler.post(r);
    }
  }

  @Override
  public void onRequestPermissionsResult(
      final int requestCode, final String[] permissions, final int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == PERMISSIONS_REQUEST) {
      if (allPermissionsGranted(grantResults)) {
        setFragment();
      } else {
        requestPermission();
      }
    }
  }

  private static boolean allPermissionsGranted(final int[] grantResults) {
    for (int result : grantResults) {
      if (result != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }
    return true;
  }

  private boolean hasPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
    } else {
      return true;
    }
  }

  private void requestPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
        Toast.makeText(
                CameraActivity.this,
                "Camera permission is required for this demo",
                Toast.LENGTH_LONG)
            .show();
      }
      requestPermissions(new String[] {PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
    }
  }

  // Returns true if the device supports the required hardware level, or better.
  private boolean isHardwareLevelSupported(
      CameraCharacteristics characteristics, int requiredLevel) {
      int deviceLevel = 0;
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
          deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
      }
      if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
      return requiredLevel == deviceLevel;
    }
    // deviceLevel is not LEGACY, can use numerical sort
    return requiredLevel <= deviceLevel;
  }

  @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
  private String chooseCamera() {
    final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
    try {
      for (final String cameraId : manager.getCameraIdList()) {
        final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

        // We don't use a front facing camera in this sample.
        final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
        if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
          continue;
        }

        final StreamConfigurationMap map =
            characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

        if (map == null) {
          continue;
        }

        // Fallback to camera1 API for internal cameras that don't have full support.
        // This should help with legacy situations where using the camera2 API causes
        // distorted or otherwise broken previews.
        useCamera2API =
            (facing == CameraCharacteristics.LENS_FACING_EXTERNAL)
                || isHardwareLevelSupported(
                    characteristics, CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL);
        LOGGER.i("Camera API lv2?: %s", useCamera2API);
        return cameraId;
      }
    } catch (CameraAccessException e) {
      LOGGER.e(e, "Not allowed to access camera");
    }

    return null;
  }

  protected void setFragment() {
    String cameraId = chooseCamera();

    Fragment fragment;
    if (useCamera2API) {
      CameraConnectionFragment camera2Fragment =
          CameraConnectionFragment.newInstance(
              new CameraConnectionFragment.ConnectionCallback() {
                @Override
                public void onPreviewSizeChosen(final Size size, final int rotation) {
                  previewHeight = size.getHeight();
                  previewWidth = size.getWidth();
                  CameraActivity.this.onPreviewSizeChosen(size, rotation);
                }
              },
              this,
              getLayoutId(),
              getDesiredPreviewFrameSize());

      camera2Fragment.setCamera(cameraId);
      fragment = camera2Fragment;
    } else {
      fragment =
          new LegacyCameraConnectionFragment(this, getLayoutId(), getDesiredPreviewFrameSize());
    }

    getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
  }

  protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes) {
    // Because of the variable row stride it's not possible to know in
    // advance the actual necessary dimensions of the yuv planes.
    for (int i = 0; i < planes.length; ++i) {
      final ByteBuffer buffer = planes[i].getBuffer();
      if (yuvBytes[i] == null) {
        LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity());
        yuvBytes[i] = new byte[buffer.capacity()];
      }
      buffer.get(yuvBytes[i]);
    }
  }

  public boolean isDebug() {
    return debug;
  }

  protected void readyForNextImage() {
    if (postInferenceCallback != null) {
      postInferenceCallback.run();
    }
  }

  protected int getScreenOrientation() {
    switch (getWindowManager().getDefaultDisplay().getRotation()) {
      case Surface.ROTATION_270:
        return 270;
      case Surface.ROTATION_180:
        return 180;
      case Surface.ROTATION_90:
        return 90;
      default:
        return 0;
    }
  }

  @Override
  public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
    setUseNNAPI(isChecked);
    if (isChecked) apiSwitchCompat.setText("NNAPI");
    else apiSwitchCompat.setText("TFLITE");
  }

  @Override
  public void onClick(View v) {
    int id = v.getId();
    
    // Xử lý nút plus/minus cho cả portrait và landscape
    if (id == R.id.btn_rotate_screen || id == R.id.btn_rotate_screen_right) {
      rotateScreen();
      
    }
  }

  /**
   * Điều chỉnh số threads
   */
  private void adjustThreads(int adjustment) {
    TextView currentThreadsView = threadsTextView != null ? threadsTextView : threadsTextViewRight;
    if (currentThreadsView == null) return;
    
    String threads = currentThreadsView.getText().toString().trim();
    int numThreads = Integer.parseInt(threads);
    numThreads += adjustment;
    
    if (numThreads < 1) numThreads = 1;
    if (numThreads > 9) numThreads = 9;
    
    // Cập nhật cả hai textviews nếu có
    if (threadsTextView != null) {
      threadsTextView.setText(String.valueOf(numThreads));
    }
    if (threadsTextViewRight != null) {
      threadsTextViewRight.setText(String.valueOf(numThreads));
    }
    
    setNumThreads(numThreads);
  }

  /**
   * Điều chỉnh zoom level cho camera
   */
  private void adjustZoom(int zoomChange) {
    currentZoomLevel += zoomChange;
    
    // Giới hạn zoom level
    if (currentZoomLevel < MIN_ZOOM) currentZoomLevel = MIN_ZOOM;
    if (currentZoomLevel > MAX_ZOOM) currentZoomLevel = MAX_ZOOM;
    
    // Cập nhật UI
    if (zoomValueTextViewRight != null) {
      zoomValueTextViewRight.setText(currentZoomLevel + "%");
    }
    
    // Áp dụng zoom cho UVC camera
    applyZoomToCamera();
    
    Toast.makeText(this, "Zoom: " + currentZoomLevel + "%", Toast.LENGTH_SHORT).show();
  }

  /**
   * Áp dụng zoom cho camera
   */
  private void applyZoomToCamera() {
    try {
      if (mUVCCamera != null) {
        // Convert percentage to UVC camera zoom (0-100%)
        int uvcZoom = (currentZoomLevel - MIN_ZOOM) * 100 / (MAX_ZOOM - MIN_ZOOM);
        mUVCCamera.setZoom(uvcZoom);
        
        // Update bounding box coordinates for zoom
        if (this instanceof DetectorActivity) {
          ((DetectorActivity) this).updateTrackerZoomFactor(currentZoomLevel);
        }
      }
    } catch (Exception e) {
      LOGGER.e(e, "Failed to set zoom");
    }
  }

  /**
   * Áp dụng auto zoom cho chế độ landscape để loại bỏ khoảng đen
   */
  private void applyLandscapeAutoZoom() {
    int orientation = getResources().getConfiguration().orientation;
    if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
      // Tự động zoom lên 150% để loại bỏ khoảng đen trong landscape
      // và fill toàn bộ màn hình
      currentZoomLevel = 150;
      
      // Cập nhật UI nếu có
      if (zoomValueTextViewRight != null) {
        zoomValueTextViewRight.setText(currentZoomLevel + "%");
      }
      
      // Delay một chút để camera được setup hoàn toàn
      new Handler(getMainLooper()).postDelayed(new Runnable() {
        @Override
        public void run() {
          applyZoomToCamera();
        }
      }, 500);
      
      LOGGER.i("Applied auto zoom %d%% for landscape mode", currentZoomLevel);
    }
  }

  /**UP
   * Toggle việc ẩn/hiện right sheet trong landscape mode
   */
  private void toggleRightSheet() {
    if (rightSheetLayout == null || rightGestureLayout == null || rightSheetArrowImageView == null) {
      return;
    }

    if (isRightSheetHidden) {
      // Hidden -> Expanded: Hiện lại sheet
      showRightSheet();
    } else {
      // Expanded -> Hidden: Ẩn hoàn toàn sheet
      hideRightSheet();
    }
  }

  /**
   * Hiển thị right sheet từ trạng thái ẩn
   */
  private void showRightSheet() {
    if (rightSheetLayout == null || rightSheetArrowImageView == null) {
      return;
    }

    // Đưa right sheet về vị trí expanded
    rightSheetLayout.animate()
        .translationX(0)
        .setDuration(300)
        .setInterpolator(new android.view.animation.DecelerateInterpolator())
        .start();
    
    rightSheetArrowImageView.setImageResource(R.drawable.ic_arrow_right);
    isRightSheetExpanded = true;
    isRightSheetHidden = false;
  }

  /**
   * Ẩn right sheet hoàn toàn khỏi màn hình
   */
  private void hideRightSheet() {
    if (rightSheetLayout == null || rightSheetArrowImageView == null) {
      return;
    }

    final int currentWidth = rightSheetLayout.getWidth();
    
    // Slide right sheet hoàn toàn ra khỏi màn hình
    rightSheetLayout.animate()
        .translationX(currentWidth)
        .setDuration(300)
        .setInterpolator(new android.view.animation.DecelerateInterpolator())
        .start();
    
    rightSheetArrowImageView.setImageResource(R.drawable.ic_arrow_left);
    isRightSheetExpanded = false;
    isRightSheetHidden = true;
    
    // Hiện toast thông báo người dùng có thể chạm để hiện lại
    Toast.makeText(this, "Chạm vào màn hình để hiện lại bảng điều khiển", Toast.LENGTH_SHORT).show();
  }

  /**
   * Khởi tạo UI dựa trên orientation
   */
  private void initializeUI() {
    int orientation = getResources().getConfiguration().orientation;
    
    if (orientation == android.content.res.Configuration.ORIENTATION_LANDSCAPE) {
      initializeLandscapeUI();
    } else {
      initializePortraitUI();
    }
  }

  /**
   * Khởi tạo UI cho chế độ portrait
   */
  private void initializePortraitUI() {
    bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
    if (bottomSheetLayout != null) {
      gestureLayout = findViewById(R.id.gesture_layout);
      sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout);
      bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow);
      rotateScreenButton = findViewById(R.id.btn_rotate_screen);
      
      // Initialize confidence threshold controls
      confidenceSeekBar = findViewById(R.id.confidence_seekbar);
      confidenceValueTextView = findViewById(R.id.confidence_value);
      setupConfidenceSeekBar();
      
      setupBottomSheet();
    }
  }

  /**
   * Khởi tạo UI cho chế độ landscape
   */
  private void initializeLandscapeUI() {
    rightSheetLayout = findViewById(R.id.right_sheet_layout);
    if (rightSheetLayout != null) {
      rotateScreenButtonRight = (ImageButton) findViewById(R.id.btn_rotate_screen_right);
      frameValueTextViewRight = findViewById(R.id.frame_info_right);
      cropValueTextViewRight = findViewById(R.id.crop_info_right);
      inferenceTimeTextViewRight = findViewById(R.id.inference_info_right);
      
      // Initialize confidence threshold controls for landscape
      confidenceSeekBarRight = findViewById(R.id.confidence_seekbar_right);
      confidenceValueTextViewRight = findViewById(R.id.confidence_value_right);
      setupConfidenceSeekBarRight();
      
      setupRightSheet();
    }
  }

  /**
   * Setup bottom sheet cho portrait mode
   */
  private void setupBottomSheet() {
    if (gestureLayout == null || sheetBehavior == null) return;
    
    ViewTreeObserver vto = gestureLayout.getViewTreeObserver();
    vto.addOnGlobalLayoutListener(
        new ViewTreeObserver.OnGlobalLayoutListener() {
          @Override
          public void onGlobalLayout() {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.JELLY_BEAN) {
              gestureLayout.getViewTreeObserver().removeGlobalOnLayoutListener(this);
            } else {
              gestureLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this);
            }
            int gestureHeight = gestureLayout.getMeasuredHeight();
            
            // Tính toán chiều cao để che phần khoảng đen ở dưới
            int screenHeight = getResources().getDisplayMetrics().heightPixels;
            
            // Tính toán peek height để che phần khoảng đen
            int additionalHeight = (int)(screenHeight * 0.08);
            int calculatedPeekHeight = gestureHeight + additionalHeight;
            
            // Đảm bảo peek height không vượt quá 35% màn hình để dễ kéo xuống
            int maxPeekHeight = (int)(screenHeight * 0.35);
            calculatedPeekHeight = Math.min(calculatedPeekHeight, maxPeekHeight);
            calculatedPeekHeight = Math.max(calculatedPeekHeight, gestureHeight);
            
            sheetBehavior.setPeekHeight(calculatedPeekHeight);
          }
        });
    sheetBehavior.setHideable(true);

    sheetBehavior.setBottomSheetCallback(
        new BottomSheetBehavior.BottomSheetCallback() {
          @Override
          public void onStateChanged(@NonNull View bottomSheet, int newState) {
            switch (newState) {
              case BottomSheetBehavior.STATE_HIDDEN:
                Toast.makeText(CameraActivity.this, "Chạm vào màn hình để hiện lại bảng điều khiển", Toast.LENGTH_SHORT).show();
                break;
              case BottomSheetBehavior.STATE_EXPANDED:
                bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_down);
                break;
              case BottomSheetBehavior.STATE_COLLAPSED:
                bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                break;
              case BottomSheetBehavior.STATE_SETTLING:
                bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                break;
            }
          }

          @Override
          public void onSlide(@NonNull View bottomSheet, float slideOffset) {}
        });
  }

  /**
   * Setup right sheet cho landscape mode
   */
  private void setupRightSheet() {
    if (rightSheetLayout == null) return;
    
    // Setup click listeners cho right sheet
    if (rotateScreenButtonRight != null) {
      rotateScreenButtonRight.setOnClickListener(this);
    }
    if (plusImageViewRight != null) {
      plusImageViewRight.setOnClickListener(this);
    }
    if (minusImageViewRight != null) {
      minusImageViewRight.setOnClickListener(this);
    }
    if (zoomInImageViewRight != null) {
      zoomInImageViewRight.setOnClickListener(this);
    }
    if (zoomOutImageViewRight != null) {
      zoomOutImageViewRight.setOnClickListener(this);
    }
    if (apiSwitchCompatRight != null) {
      apiSwitchCompatRight.setOnCheckedChangeListener(this);
    }
    
    // Setup click listeners để toggle right sheet
    if (rightGestureLayout != null) {
      rightGestureLayout.setOnClickListener(v -> toggleRightSheet());
    }
    if (rightSheetArrowImageView != null) {
      rightSheetArrowImageView.setOnClickListener(v -> toggleRightSheet());
    }
    
    // Setup processor spinner cho right sheet
    if (processorSpinnerRight != null) {
      processorSpinnerRight.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
        @Override
        public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
          onProcessorChanged(ProcessorType.fromValue(position));
        }

        @Override
        public void onNothingSelected(AdapterView<?> parent) {
        }
      });
    }
    
    // Sync giá trị từ portrait controls
    syncValuesToRightSheet();
  }

  /**
   * Đồng bộ giá trị từ portrait controls sang right sheet
   */
  private void syncValuesToRightSheet() {
    if (threadsTextView != null && threadsTextViewRight != null) {
      threadsTextViewRight.setText(threadsTextView.getText());
    }
    if (apiSwitchCompat != null && apiSwitchCompatRight != null) {
      apiSwitchCompatRight.setChecked(apiSwitchCompat.isChecked());
    }
    if (zoomValueTextViewRight != null) {
      zoomValueTextViewRight.setText(currentZoomLevel + "%");
    }
    
    // Sync confidence values
    if (confidenceSeekBar != null && confidenceSeekBarRight != null) {
      confidenceSeekBarRight.setProgress(confidenceSeekBar.getProgress());
    }
    if (confidenceValueTextView != null && confidenceValueTextViewRight != null) {
      confidenceValueTextViewRight.setText(confidenceValueTextView.getText());
    }
  }

  /**
   * Setup confidence threshold SeekBar for portrait mode
   */
  private void setupConfidenceSeekBar() {
    if (confidenceSeekBar == null || confidenceValueTextView == null) return;
    
    // Set initial values
    int initialProgress = (int) (((DetectorActivity) this).getMinimumConfidence() * 100);
    confidenceSeekBar.setProgress(initialProgress);
    confidenceValueTextView.setText(String.format("%.2f", ((DetectorActivity) this).getMinimumConfidence()));
    
    confidenceSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
      @Override
      public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
        if (fromUser) {
          float confidence = progress / 100.0f;
          confidenceValueTextView.setText(String.format("%.2f", confidence));
          ((DetectorActivity) CameraActivity.this).setMinimumConfidence(confidence);
          
          // Sync with landscape SeekBar if available
          if (confidenceSeekBarRight != null && confidenceValueTextViewRight != null) {
            confidenceSeekBarRight.setProgress(progress);
            confidenceValueTextViewRight.setText(String.format("%.2f", confidence));
          }
        }
      }
      
      @Override
      public void onStartTrackingTouch(SeekBar seekBar) {}
      
      @Override
      public void onStopTrackingTouch(SeekBar seekBar) {}
    });
  }

  /**
   * Setup confidence threshold SeekBar for landscape mode
   */
  private void setupConfidenceSeekBarRight() {
    if (confidenceSeekBarRight == null || confidenceValueTextViewRight == null) return;
    
    // Set initial values
    int initialProgress = (int) (((DetectorActivity) this).getMinimumConfidence() * 100);
    confidenceSeekBarRight.setProgress(initialProgress);
    confidenceValueTextViewRight.setText(String.format("%.2f", ((DetectorActivity) this).getMinimumConfidence()));
    
    confidenceSeekBarRight.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
      @Override
      public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
        if (fromUser) {
          float confidence = progress / 100.0f;
          confidenceValueTextViewRight.setText(String.format("%.2f", confidence));
          ((DetectorActivity) CameraActivity.this).setMinimumConfidence(confidence);
          
          // Sync with portrait SeekBar if available
          if (confidenceSeekBar != null && confidenceValueTextView != null) {
            confidenceSeekBar.setProgress(progress);
            confidenceValueTextView.setText(String.format("%.2f", confidence));
          }
        }
      }
      
      @Override
      public void onStartTrackingTouch(SeekBar seekBar) {}
      
      @Override
      public void onStopTrackingTouch(SeekBar seekBar) {}
    });
  }

  /**
   * Quay màn hình giữa portrait và landscape
   */
  private void rotateScreen() {
    int currentOrientation = getRequestedOrientation();
    
    if (currentOrientation == ActivityInfo.SCREEN_ORIENTATION_PORTRAIT ||
        currentOrientation == ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED) {
      // Chuyển sang landscape
      setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
      Toast.makeText(this, "Chuyển sang chế độ ngang", Toast.LENGTH_SHORT).show();
    } else {
      // Chuyển về portrait
      setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
      Toast.makeText(this, "Chuyển sang chế độ dọc", Toast.LENGTH_SHORT).show();
    }
  }

  /**
   * Toggle bottom sheet giữa các trạng thái
   */
  private void toggleBottomSheet() {
    if (sheetBehavior.getState() == BottomSheetBehavior.STATE_HIDDEN) {
      sheetBehavior.setState(BottomSheetBehavior.STATE_COLLAPSED);
    } else if (sheetBehavior.getState() == BottomSheetBehavior.STATE_COLLAPSED) {
      sheetBehavior.setState(BottomSheetBehavior.STATE_EXPANDED);
    } else {
      sheetBehavior.setState(BottomSheetBehavior.STATE_COLLAPSED);
    }
  }

  protected void showFrameInfo(String frameInfo) {
    if (frameValueTextView != null) {
      frameValueTextView.setText(frameInfo);
    }
    if (frameValueTextViewRight != null) {
      frameValueTextViewRight.setText(frameInfo);
    }
  }

  protected void showCropInfo(String cropInfo) {
    if (cropValueTextView != null) {
      cropValueTextView.setText(cropInfo);
    }
    if (cropValueTextViewRight != null) {
      cropValueTextViewRight.setText(cropInfo);
    }
  }

  protected void showInference(String inferenceTime) {
    if (inferenceTimeTextView != null) {
      inferenceTimeTextView.setText(inferenceTime);
    }
    if (inferenceTimeTextViewRight != null) {
      inferenceTimeTextViewRight.setText(inferenceTime);
    }
  }
  protected void onProcessorChanged(ProcessorType processorType) {
    setProcessorType(processorType);
  }

  protected abstract void processImage();

  protected abstract void onPreviewSizeChosen(final Size size, final int rotation);

  protected abstract int getLayoutId();

  protected abstract Size getDesiredPreviewFrameSize();

  protected abstract void setNumThreads(int numThreads);

  protected abstract void setUseNNAPI(boolean isChecked);

  protected abstract void setProcessorType(ProcessorType processorType);

  // Biến để theo dõi trạng thái navigation bar
  private boolean isSystemUIHidden = false;


  /**
   * Ẩn system UI (navigation bar và status bar)
   */
  private void hideSystemUI() {
    View decorView = getWindow().getDecorView();
    int uiOptions = View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
            | View.SYSTEM_UI_FLAG_FULLSCREEN
            | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
            | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
            | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
            | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
            | View.SYSTEM_UI_FLAG_LOW_PROFILE; // Additional flag to minimize system UI
    decorView.setSystemUiVisibility(uiOptions);
    getWindow().setNavigationBarColor(Color.TRANSPARENT);
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
    getWindow().setStatusBarColor(Color.TRANSPARENT);
    // FORCE remove all padding/margin from system UI
    isSystemUIHidden = true;

    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
      getWindow().setNavigationBarColor(Color.TRANSPARENT);
      getWindow().setStatusBarColor(Color.TRANSPARENT);

      // Additional flags to force layout beyond system UI (TabSpec approach)
      getWindow().addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
      getWindow().clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS);
      getWindow().clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_NAVIGATION);
    }
  }


  /**
   * Hiển thị system UI (navigation bar và status bar)
   */
  private void showSystemUI() {
    View decorView = getWindow().getDecorView();
    decorView.setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                                  | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                                  | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN);
    isSystemUIHidden = false;
    LOGGER.d("System UI shown (navigation bar and status bar)");
    
    // Tự động ẩn lại navigation bar sau 3 giây
    new Handler(getMainLooper()).postDelayed(new Runnable() {
      @Override
      public void run() {
        if (!isSystemUIHidden) {
          hideSystemUI();
        }
      }
    }, 3000); // 3 giây
  }

  @Override
  public void onWindowFocusChanged(boolean hasFocus) {
    super.onWindowFocusChanged(hasFocus);
    if (hasFocus && isSystemUIHidden) {
      // Chỉ ẩn lại navigation bar nếu hiện tại đang ở trạng thái ẩn
      hideSystemUI();
    }
  }
}
