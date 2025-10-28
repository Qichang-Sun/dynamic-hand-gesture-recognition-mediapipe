# 创建虚拟环境
# py -3.11 -m venv venv-mp311

# 激活虚拟环境
#.\venv-mp311\Scripts\Activate.ps1

# 退出虚拟环境
# deactivate

# Run these 2 commands below after install python 3.11

# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# powershell -ExecutionPolicy Bypass -File .\setup-mediapipe-env.ps1


# setup-mediapipe-env.ps1
param(
    [string]$EnvName = "venv-mp311"
)

Write-Host "=== Checking installed Python versions ==="
py --version

Write-Host "`n=== Creating and activating Python 3.11 virtual environment ==="
py -3.11 -m venv $EnvName
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create venv with Python 3.11."
    exit 1
}
. .\$EnvName\Scripts\Activate.ps1

Write-Host "`n=== Upgrading pip/setuptools/wheel ==="
python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upgrade pip/setuptools/wheel."
    exit 1
}

Write-Host "`n=== Installing dependencies ==="
# Pin protobuf 3.20.* for old mediapipe compatibility
python -m pip install "protobuf==3.20.*"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install protobuf==3.20.*"
    exit 1
}

# Install mediapipe & others
python -m pip install mediapipe==0.8.10.1 opencv-python>=4.5.0 tensorflow==2.5.0 scikit-learn>=0.23.2 matplotlib>=3.3.2
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Installing mediapipe==0.8.10.1 failed. Trying mediapipe (latest) as fallback ..."
    python -m pip install mediapipe
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install mediapipe. Please ensure you're in Python 3.11 x64."
        exit 1
    }
}

Write-Host "`n=== Verifying versions ==="
$verify = @'
import sys
print("Python:", sys.version.replace("\n"," "))
try:
    import mediapipe as mp
    print("MediaPipe:", mp.__version__)
except Exception as e:
    print("MediaPipe: ERROR ->", e)

try:
    import cv2
    print("OpenCV:", cv2.__version__)
except Exception as e:
    print("OpenCV: ERROR ->", e)

try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
except Exception as e:
    print("TensorFlow: ERROR ->", e)

try:
    import sklearn
    print("scikit-learn:", sklearn.__version__)
except Exception as e:
    print("scikit-learn: ERROR ->", e)

try:
    import matplotlib
    print("matplotlib:", matplotlib.__version__)
except Exception as e:
    print("matplotlib: ERROR ->", e)
'@
$verify | python -

Write-Host "`n✅ Environment is ready and activated: $EnvName"
Write-Host "To exit the virtual environment: deactivate"
