@echo off
set JAVA_HOME=C:\Program Files (x86)\Java\jdk-1.8
set PATH=%JAVA_HOME%\bin;%PATH%

echo Using JAVA_HOME=%JAVA_HOME%

echo [2/3] Building APK...
call gradlew assembleDebug

echo [3/3] Installing APK to device...
adb install -r "D:\download\objectDetection\TFObjectDetectionDemoApp-master -90_TestModelAnhHieu_moblienetSSD7\app\build\outputs\apk\debug\app-debug.apk"

echo.
echo ✅ Done! APK installed successfully.
pause
