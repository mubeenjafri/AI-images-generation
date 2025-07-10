AI Image Generator - Setup and Usage Guide
===============================================

This tool generates AI images using OpenAI's image generation models based on prompts from a CSV file.

You can choose between two models:
- **DALL-E 3**: Widely available, high-quality image generation (recommended for most users)
- **GPT Image 1**: Advanced model with superior instruction following (requires verified organization)

QUICK START (Windows)
--------------------
1. Double-click "run.bat" - it will handle everything automatically
2. Choose your preferred AI model (DALL-E 3 or GPT Image 1)
3. Your browser will open showing progress
4. Generated images will be saved in the "output" folder

MANUAL SETUP (if run.bat doesn't work)
-------------------------------------

Prerequisites:
- Python 3.8 or higher
- OpenAI API key (get one from https://platform.openai.com/api-keys)

Step 1: Install Python
- Download from https://www.python.org/downloads/
- During installation, CHECK "Add Python to PATH"
- Restart your computer after installation

Step 2: Verify Python Installation
Open Command Prompt (cmd) or PowerShell and run:
  python --version

You should see something like "Python 3.11.0"

Step 3: Create .env File
Create a file named ".env" (no extension) in the same folder as this README.
Add your OpenAI API key:
  OPENAI_API_KEY=sk-your-actual-key-here

Step 4: Prepare CSV File
Make sure "prompts.csv" exists with these columns:
  prompt,filename,orientation,status

Example:
  prompt,filename,orientation,status
  A beautiful sunset,image001,square,
  A futuristic city,image002,landscape,

Step 5: Setup Virtual Environment
Open Command Prompt in this folder and run:
  python -m venv venv
  venv\Scripts\activate.bat

Step 6: Install Dependencies
With the virtual environment activated, run:
  pip install -r requirements.txt

Step 7: Run the Script
  python generate_images.py --csv prompts.csv --model dall-e-3

COMMAND LINE OPTIONS
-------------------
--csv      Path to your CSV file (required)
--model    AI model to use: "dall-e-3" or "gpt-image-1" (default: dall-e-3)
--out      Output folder for images (default: output)
--n        Number of images per prompt, 1-10 (default: 1)
--port     Web server port (default: 5000)

Examples:
  python generate_images.py --csv my_prompts.csv --model gpt-image-1 --n 3
  python generate_images.py --csv prompts.csv --model dall-e-3 --out my_images

MULTIPLE IMAGES PER PROMPT
--------------------------

Filename convention:
- If n=1: filename becomes "image001.png"
- If n=3: filenames become "image001_1.png", "image001_2.png", "image001_3.png"

CSV FILE FORMAT
--------------
Required columns:
- prompt: Text description for the image
- filename: Base name for the output file (without .png)
- orientation: "square", "portrait", "landscape" (or legacy: "vertical", "horizontal")
- status: Leave empty initially, script will update this

The script will:
- Skip rows where status = "done"
- Mark completed rows as "done"
- Mark failed rows as "error"
- Update the CSV file after each run

Note: If you generate 3 images per prompt, the status is marked "done" only 
after ALL 3 images are successfully created.

IMAGE SIZES AND ORIENTATIONS
----------------------------

DALL-E 3 supported sizes:
- square: 1024x1024 pixels
- portrait: 1024x1792 pixels (vertical)
- landscape: 1792x1024 pixels (horizontal)

GPT Image 1 supported sizes:
- square: 1024x1024 pixels
- portrait: 1024x1536 pixels (vertical)
- landscape: 1536x1024 pixels (horizontal)

Legacy orientation names are still supported:
- vertical: same as portrait
- horizontal: same as landscape

All images are generated in PNG format.