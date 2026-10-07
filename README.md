# ISL-Sign-to-speech-conversion
ISL Sign-to-Speech Module is a computer-vision and deep-learning based system designed to recognize Indian Sign Language (ISL) hand signs through a camera and convert recognized signs into text.
The current implementation focuses on ISL alphabet recognition (A–Z). Recognized letters are accumulated to form words, which can then be converted into speech using a Text-to-Speech system.

Conversion Flow


[ Camera ]
    │
    ▼
[ Hand Detection ]
    │
    ▼
[ Hand Region Extraction ]
    │
    ▼
[ Image Preprocessing ]
    │
    ▼
[ ResNet18 CNN ]
    │
    ▼
[ ISL Alphabet Recognition ]
    │
    ▼
[ Prediction Stabilization ]
    │
    ▼
[ Letter Sequence ]
    │
    ▼
[ Word Formation ]
    │
    ▼
[ Text ]
    │
    ▼
[ Text-to-Speech ]

🎯 What Is This Module?

This module converts visual Indian Sign Language gestures into computer-readable text.

For example:

User performs ISL signs

H → E → L → L → O

        ↓

     "HELLO"

        ↓

 Text-to-Speech

        ↓

     🔊 HELLO

The camera continuously captures the user's hand gesture. The system detects the hand, extracts the relevant region, and sends the image to a trained deep-learning model.

The model predicts the corresponding ISL alphabet character.

Multiple recognized characters are then combined to create words.

🧠 How Does It Work?
1. Camera Input

The webcam captures frames continuously.

Webcam
   ↓
Video Frames

OpenCV is used to access and process the camera stream.

2. Hand Detection

The system uses MediaPipe Hand Landmarker to locate the user's hand.

The detector identifies the hand region and provides information about the hand's position.

Camera Frame
      ↓
MediaPipe
      ↓
Hand Region

This reduces the amount of unnecessary information given to the classification model.

3. Image Preprocessing

The detected hand region is extracted and converted into a suitable format for the neural network.

Typical preprocessing includes:

Hand-region cropping
Square cropping
Image resizing
RGB conversion
Tensor conversion
Pixel normalization

The processed image is then passed to the classification model.

🤖 Recognition Model
ResNet18 CNN

The main classification model is ResNet18.

ResNet18 is a deep Convolutional Neural Network (CNN) architecture designed for image classification.

In this project, the final classification layer is configured for:

26 Classes
A → Z
Model Pipeline
Input Image
     ↓
Convolution Layers
     ↓
Residual Blocks
     ↓
Feature Extraction
     ↓
Fully Connected Layer
     ↓
26 ISL Classes
     ↓
Predicted Letter

Why ResNet18?

ResNet18 was selected because it provides a good balance between:

Recognition accuracy
Computational requirements
Training speed
Inference speed
Suitability for future mobile deployment

The relatively lightweight architecture also makes it more suitable for eventual integration into an Android application than very large vision models.

📊 Dataset
RealSign — Indian Sign Language Dataset

The project uses the RealSign Indian Sign Language Dataset for ISL alphabet training.

Repository:

RealSign62/RealSign-Indian-Sign-Language-Dataset

The dataset contains ISL alphabet classes and is organized for machine-learning training and evaluation.

The current training configuration used in this project contains approximately:

Training images     : 18,198
Validation images   : 2,579
Testing images      : 5,200
Classes             : 26
Classes
A B C D E F G I J K L M N O P Q R S T U V W X Y Z

Dataset licensing and usage conditions should be checked in the original dataset repository before redistribution or commercial deployment.

🧪 Training

The model is trained using supervised learning.

Each training image has an associated ISL alphabet label.

Example:

Image → A
Image → B
Image → C
...
Image → Z

During training:

Training Image
      ↓
ResNet18
      ↓
Prediction
      ↓
Compare with Correct Label
      ↓
Calculate Loss
      ↓
Backpropagation
      ↓
Update Model Weights

The process is repeated over multiple epochs.

Example:

python train.py --epochs 15
⚡ GPU Acceleration

GPU acceleration is strongly recommended for training.

The project supports CUDA-enabled PyTorch.

Check GPU availability:

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU not detected')"

Expected output on a supported NVIDIA system:

True
NVIDIA GeForce RTX 3050

If CUDA is unavailable, PyTorch falls back to CPU:

Device: cpu

Training on CPU can be significantly slower for a large image dataset.

🔄 Prediction Stabilization

A camera-based recognition system should not accept every individual frame as a new letter.

Without stabilization:

Frame 1 → A
Frame 2 → B
Frame 3 → A
Frame 4 → C
Frame 5 → A

This creates incorrect text.

The module therefore uses consecutive predictions to determine whether a gesture is stable.

Example:

A
A
A
A
A
 ↓
Accept A

This reduces random predictions and improves practical webcam stability.

✍️ Word Formation

The current classifier recognizes individual alphabet signs.

The word-building layer combines recognized letters.

Example:

H
E
L
L
O

becomes:

HELLO

A space can be inserted to separate words:

HELLO WORLD

The text buffer can also support operations such as:

Add letter
Add space
Delete last character
Clear text
Speak current text
🔊 Text-to-Speech

After a word or sentence is generated, the text can be converted into speech.

ISL Gesture
     ↓
Letter
     ↓
Word
     ↓
Sentence
     ↓
Text-to-Speech
     ↓
Audio Output

This allows a person who does not understand ISL to hear the converted communication.

🛠️ Technologies Used
Technology	Purpose
Python	Main development language
PyTorch	Deep-learning framework
TorchVision	Computer-vision datasets, transforms and models
ResNet18	ISL image classification
OpenCV	Camera access and image processing
MediaPipe	Hand detection/landmark processing
NumPy	Numerical/image processing
Pillow	Image processing
Text-to-Speech	Converts recognized text into speech
📦 Main Python Libraries

Install the project dependencies using:

pip install -r requirements.txt

Major dependencies include:

torch
torchvision
opencv-python
mediapipe
numpy
Pillow

Additional dependencies may be required depending on the Text-to-Speech implementation.

💻 System Requirements
Minimum
Windows 10/11
Python 3.10 or 3.11
8 GB RAM
Webcam
CPU capable of running PyTorch
Sufficient disk space for the dataset
Recommended
Windows 10/11
Python 3.11
16 GB RAM
NVIDIA GPU
4 GB or more VRAM
RTX 3050 or better
HD webcam
SSD storage
🐍 Python Environment

A virtual environment is recommended.

Create:

python -m venv .venv

Activate on Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Then install dependencies:

python -m pip install --upgrade pip
pip install -r requirements.txt

Using a virtual environment prevents dependency conflicts with other Python projects.

📁 Recommended Project Structure
NexusCue-ISL/
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   └── isl_resnet18.pth
│
├── main.py
├── train.py
├── model.py
├── config.py
├── prepare_dataset.py
├── download_hand_model.py
├── requirements.txt
├── README.md
└── .gitignore
🚀 Installation
1. Clone the repository
git clone <repository-url>
cd NexusCue-ISL
2. Create virtual environment
python -m venv .venv
3. Activate environment
.\.venv\Scripts\Activate.ps1
4. Install dependencies
pip install -r requirements.txt
5. Prepare the dataset

Download the RealSign ISL dataset from:

RealSign ISL Dataset

Place or prepare the dataset according to the project's expected directory structure.

🏋️ Training the Model

Run:

python train.py --epochs 15

For a quick test:

python train.py --epochs 2

Training produces a trained model checkpoint:

models/isl_resnet18.pth
🎥 Running Real-Time Recognition

After training:

python main.py

The webcam will open and begin recognizing ISL alphabet signs.

Typical controls:

SPACE       Add space
BACKSPACE   Delete last character
C           Clear text
S           Speak text
Q           Quit
📈 Accuracy

Training accuracy and validation accuracy should be monitored during training.

The final practical accuracy depends on:

Dataset quality
Lighting conditions
Camera quality
Hand position
Background
Sign orientation
Distance from camera
Training configuration
Number of training epochs
GPU/CPU configuration

A high validation accuracy does not guarantee identical accuracy in real-world camera conditions.

⚠️ Current Limitations

The current implementation is primarily an ISL alphabet recognition system.

It should not be described as a complete continuous ISL translation system yet.

Current limitations

1. Alphabet-based recognition

The current ResNet18 model recognizes individual alphabet signs.

A → Z

It does not directly understand arbitrary ISL sentences.

2. Dynamic signs

Some signs involve movement over time. A static image classifier cannot reliably represent every dynamic sign.

3. Word recognition

Words are currently constructed from recognized letters.

H + E + L + L + O
          ↓
        HELLO

This is different from directly recognizing the complete ISL word gesture.

4. Real-world conditions

Accuracy may decrease with:

Poor lighting
Occlusion
Multiple hands
Complex backgrounds
Unusual camera angles
Very fast movement





