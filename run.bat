@echo off

echo ==========================================
echo   AI Image Generator Setup and Runner
echo ==========================================
echo.

REM Check if Python is installed
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

python --version
echo Python found successfully.
echo.

REM Check if .env file exists
echo [2/7] Checking for .env file...
if not exist ".env" (
    echo ERROR: .env file not found.
    echo.
    echo Please create a .env file in this directory with your OpenAI API key:
    echo OPENAI_API_KEY=sk-your-key-here
    echo.
    pause
    exit /b 1
)
echo .env file found.
echo.

REM Check if CSV file exists
echo [3/7] Checking for CSV file...
if not exist "prompts.csv" (
    echo ERROR: prompts.csv not found.
    echo.
    echo Please ensure prompts.csv exists in this directory with columns:
    echo prompt,filename,orientation,status
    echo.
    pause
    exit /b 1
)
echo prompts.csv found.
echo.

REM Get user input for number of images per prompt
echo [4/7] Configuration...
echo.

REM Ask user to select model
echo Which AI model would you like to use?
echo.
echo 1. DALL-E 3 (Recommended - works for all users)
echo    - High-quality image generation
echo    - Widely available
echo    - Reliable and tested
echo.
echo 2. GPT Image 1 (Advanced - requires verified organization)
echo    - Superior instruction following
echo    - Better world knowledge
echo    - Requires organization verification on OpenAI
echo.

REM --- model choice ------------------------------------------------
set "model_choice="
set /p model_choice="Enter your choice (1 or 2, default=1): "
if "%model_choice%"=="" set "model_choice=1"

REM trim quotes and spaces
for /f "tokens=* delims= " %%A in ("%model_choice:"=%") do set "model_choice=%%A"

if "%model_choice%"=="1" goto :model_dalle3
if "%model_choice%"=="2" goto :model_gpt_image1

echo ERROR: Please enter 1 or 2.
pause
exit /b 1

:model_dalle3
set "selected_model=dall-e-3"
echo Selected: DALL-E 3
goto :model_selected

:model_gpt_image1
set "selected_model=gpt-image-1"
echo Selected: GPT Image 1
echo.
echo NOTE: If you get an "organization must be verified" error,
echo please choose option 1 (DALL-E 3) instead.
goto :model_selected
echo.

:model_selected
echo.

set images_per_prompt=1
set /p images_per_prompt="How many images do you want to generate per prompt? (1-10, default=1): "
if "%images_per_prompt%"=="" set images_per_prompt=1

REM Validate input - check if it's a number between 1 and 10
REM First, remove any quotes from the input
set images_per_prompt=%images_per_prompt:"=%

REM Try to convert to number
set /a test_num=%images_per_prompt% 2>nul

REM Check if conversion was successful and the result is valid
if %test_num% geq 1 if %test_num% leq 10 (
    set images_per_prompt=%test_num%
    goto :number_valid
)

REM If we get here, the input was invalid
echo ERROR: Please enter a valid number between 1 and 10.
pause
exit /b 1

:number_valid
echo Will generate %images_per_prompt% image(s) per prompt.
echo.

REM Create and activate virtual environment
echo [5/7] Setting up virtual environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        echo Make sure you have sufficient permissions and disk space.
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    echo The venv folder may be corrupted. Try deleting it and running again.
    pause
    exit /b 1
)
echo Virtual environment activated.
echo.

REM Install dependencies
echo [6/7] Installing dependencies...
echo This may take a moment...
pip install -r requirements.txt --quiet --disable-pip-version-check
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    echo.
    echo This could be due to:
    echo - Network connectivity issues
    echo - Missing requirements.txt file
    echo - Insufficient permissions
    echo.
    echo Try running this command manually:
    echo pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo Dependencies installed successfully.
echo.

REM Run the script
echo [7/7] Starting AI image generation...
echo.
echo Configuration:
echo - CSV file: prompts.csv
echo - Model: %selected_model%
echo - Images per prompt: %images_per_prompt%
echo - Output folder: output
echo.
echo Browser will open automatically to show progress.
echo Press Ctrl+C to stop the script when completed.
echo.

python generate_images.py --csv prompts.csv --model %selected_model% --n %images_per_prompt%
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Script execution failed.
    echo.
    echo Common issues:
    echo - Invalid OpenAI API key in .env file
    echo - No internet connection
    echo - OpenAI API service issues
    echo - Invalid prompts in CSV file
    echo.
    echo Check the error messages above for specific details.
    echo.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Script completed successfully!
echo.
echo Generated images are saved in the 'output' folder.
echo The CSV file has been updated with completion status.
echo.
echo You can run this script again to:
echo - Generate images for new prompts
echo - Retry failed prompts
echo - Skip already completed prompts
echo ==========================================
pause 