# Hand Gesture Recognition 🤖✋

This project uses **MediaPipe** and a **Neural Network** to recognize hand gestures in real-time using a webcam 🎥. It detects gestures like:

- "Open" ✋
- "Close" ✊
- "Thumbs Up" 👍
- "F**k Off" 🤟

You can also record more data :)

## Installation 📦

1. Clone the repository:
   ```bash
   git clone https://github.com/SwarritSrivastava/Neural_Network_L3-numpy-Only
   ```
2. Install the required dependencies:
   ```bash
   pip install mediapipe opencv-python numpy
   ```
## Usage 🚀
                                          
1. Run the script to start recognizing gestures:
   ```bash
   python RUN.py
   ```
2. The script will open a webcam feed and display the predicted gesture on the screen.

## How it Works 🔧

1. **MediaPipe** detects hand landmarks in real-time from the webcam feed.
2. The landmarks are processed and passed through a **THREE-LAYER NEURAL NETWORK** for gesture classification.
3. The predicted gesture is displayed on the screen in real-time.

## Additional Features

1. **To record more data:**
   Run:
   ```bash
   python MakeData.py
   ```
   and follow the given instructions.
2. **Edit the Neural Network:**
   Open `Neural_Network_L3.py` with a code editor and make the necessary changes.
   
## Model Weights 🏅
The model is trained on hand gesture data. You can find the trained weights in the repository, or train your own model by following the training process in the code.

## Contributing 🤝
Feel free to fork the project, open issues, and submit pull requests. Contributions are welcome!

Made with <3 by Swarit Srivastava

### Sections Explained:

- **Installation**: Guides the user to clone the repo and install necessary libraries.
- **Usage**: Explains how to run the code.
- **How it works**: Brief description of the working logic.
- **Model Weights**: Mentions where the model weights are located.
- **Contributing**: Invites others to contribute to the project.
- **License**: License information for the project.
