@echo off
echo ========================================
echo 📥 CHEST X-RAY DATASET DOWNLOADER
echo ========================================
echo.
echo This script will help you download the dataset
echo.
echo STEP 1: Open your browser and go to:
echo https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
echo.
echo STEP 2: Click the "Download" button
echo STEP 3: Save the file as "chest-xray-pneumonia.zip"
echo.
echo STEP 4: Extract the ZIP file to:
echo %CD%\data\chest_xray\
echo.
echo The download is approximately 1.2GB
echo.
echo Press any key to open the download page...
pause >nul

start https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

echo.
echo After downloading and extracting:
echo Run: python train_model.py
echo.
pause