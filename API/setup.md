SET PATH=%PATH%;C:\Users\sajin\AppData\Local\Programs\Python\Python39;C:\ffmpeg\bin;

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Linux/macOS
.\venv\Scripts\activate   # On Windows

pip install -r requirements.txt

# Install FastAPI and Uvicorn
pip install fastapi uvicorn