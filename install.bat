@echo off
REM DocParser Installation Script for Windows
REM This script sets up the DocParser environment using conda

echo === DocParser Installation Script ===
echo.

REM Check if conda is installed
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Conda is not installed. Please install Anaconda or Miniconda first.
    echo    Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo ✅ Conda found

REM Check if environment already exists
conda env list | findstr "docparser" >nul
if %errorlevel% equ 0 (
    echo ⚠️  DocParser environment already exists.
    set /p recreate="Do you want to recreate it? (y/N): "
    if /i "%recreate%"=="y" (
        echo Removing existing environment...
        conda env remove -n docparser -y
    ) else (
        echo Using existing environment.
        call conda activate docparser
        echo ✅ Environment activated.
        echo Run: python verify_installation.py
        pause
        exit /b 0
    )
)

REM Create environment from yml file
echo Creating DocParser environment...
if exist "environment.yml" (
    conda env create -f environment.yml
) else (
    echo environment.yml not found, creating basic environment...
    conda create -n docparser python=3.9 -y
    call conda activate docparser
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    conda install -c conda-forge pillow numpy scikit-learn tqdm -y
    pip install transformers>=4.30.0 PyMuPDF>=1.23.0 pytesseract>=0.3.10 opencv-python>=4.8.0 scikit-image>=0.21.0
)

REM Activate environment
echo Activating environment...
call conda activate docparser

REM Install the package in development mode
echo Installing DocParser in development mode...
pip install -e .

echo.
echo ✅ Installation completed!
echo.
echo Next steps:
echo 1. Verify installation: python verify_installation.py
echo 2. Try basic usage: python examples/basic_usage.py
echo 3. Check documentation in README.md
echo.
echo To activate the environment in the future:
echo    conda activate docparser
pause 