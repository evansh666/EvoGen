# Check if gdown is installed, and install it if not
if ! command -v gdown &> /dev/null; then
    pip install gdown
fi

# Download the folder from Google Drive
gdown https://drive.google.com/drive/folders/1gd7k8NNRFZi4p00iwTw3LAWytXGstmQD?usp=sharing