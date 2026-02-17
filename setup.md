SET PATH=%PATH%;C:\Users\sajin\AppData\Local\Python\pythoncore-3.14-64;C:\ffmpeg\bin;
## Conda, ffmpeg,rust. Install Conda, Rust and download ffmpeg
SET PATH=%PATH%;C:\Users\sajin\anaconda3\Scripts;C:\ffmpeg\bin;C:\Users\sajin\.cargo\bin;

# Create a virtual environment
python -m venv venv OR
py -3.14 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Linux/macOS
.\venv\Scripts\activate   # On Windows

pip install -r requirements.txt
conda install -c conda-forge transformers=4.11.3

python -m pip install --upgrade pip setuptools wheel maturin

python -m streamlit run langchaintest.py

# Install FastAPI and Uvicorn
pip install fastapi uvicorn

Torch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Conda
conda init
conda create --name myenv1 python=3.11
conda activate myenv1

conda install -c conda-forge transformers
conda install pytorch torchvision torchaudio -c pytorch

streamlit run langchaintest.py

# requirements.txt removed
