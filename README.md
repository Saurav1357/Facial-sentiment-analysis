# Facial-sentiment-analysis
A real-time facial emotion detection system built with Python, OpenCV, and TensorFlow. Uses a custom CNN model to classify 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral) from live camera feed with face detection capabilities.
# Facial Sentiment Analysis

A real-time emotion detection system that uses computer vision and deep learning to recognize facial expressions and classify emotions from live camera feed or images.

## 🎯 Project Overview

This project implements a Convolutional Neural Network (CNN) to detect and classify human emotions from facial expressions. The system can identify 7 different emotions: **Angry**, **Disgust**, **Fear**, **Happy**, **Sad**, **Surprise**, and **Neutral**.

## ✨ Features

- **Real-time emotion detection** from webcam feed
- **7 emotion classifications** with high accuracy
- **Face detection** using OpenCV Haar Cascades
- **Custom CNN model** trained on facial expression datasets
- **Easy-to-use interface** with live video display
- **Preprocessed data pipeline** for training

## 🛠️ Technologies Used

- **Python 3.x**
- **OpenCV** - Computer vision and image processing
- **Keras/TensorFlow** - Deep learning framework
- **NumPy** - Numerical computations
- **scikit-learn** - Data preprocessing and splitting

## 📁 Project Structure

```
facial_sentiment_analysis/
├── prepare_data_from_images.py    # Data preprocessing script
├── train_model.py                 # Model training script
├── real_time_emotion.py          # Real-time emotion detection
├── emotion_data_from_images.npz  # Preprocessed training data
├── emotion_model.h5              # Trained CNN model
├── train/                        # Training images folder
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/                         # Testing images folder
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

## 🚀 Getting Started

### Prerequisites

Install the required packages:

```bash
pip install opencv-python
pip install tensorflow
pip install keras
pip install numpy
pip install scikit-learn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/facial_sentiment_analysis.git
cd facial_sentiment_analysis
```

2. Ensure you have the required directory structure with training and testing images organized by emotion labels.

## 💻 Usage

### 1. Data Preparation
First, prepare your dataset by running the data preprocessing script:

```bash
python prepare_data_from_images.py
```

This script will:
- Load images from `train/` and `test/` folders
- Resize images to 48x48 pixels
- Normalize pixel values
- Convert labels to categorical format
- Save processed data as `emotion_data_from_images.npz`

### 2. Model Training
Train the CNN model:

```bash
python train_model.py
```

This will:
- Load the preprocessed data
- Create and compile a CNN model
- Train for 20 epochs
- Save the trained model as `emotion_model.h5`

### 3. Real-time Emotion Detection
Run the real-time emotion detection:

```bash
python real_time_emotion.py
```

- The application will open your default camera
- Detected faces will be highlighted with rectangles
- Emotion predictions will be displayed above each face
- Press 'q' to quit the application

## 🧠 Model Architecture

The CNN model consists of:

- **Input Layer**: 48x48 grayscale images
- **Conv2D Layer**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Conv2D Layer**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Flatten Layer**: Convert to 1D
- **Dense Layer**: 128 neurons, ReLU activation
- **Dropout**: 0.5 rate for regularization
- **Output Layer**: 7 neurons, Softmax activation

## 📊 Emotion Classes

| Label | Emotion |
|-------|---------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

## 🔧 Customization

### Modify Emotions
To add or remove emotions, update the `emotion_labels` dictionary in `prepare_data_from_images.py`:

```python
emotion_labels = {
    'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
    'sad': 4, 'surprise': 5, 'neutral': 6
}
```

### Adjust Model Parameters
Modify hyperparameters in `train_model.py`:
- Change number of epochs
- Adjust batch size
- Modify learning rate
- Add more layers

## 📈 Performance Tips

- **Dataset Quality**: Use high-quality, diverse facial expression datasets
- **Data Augmentation**: Consider adding rotation, brightness, and contrast variations
- **Model Tuning**: Experiment with different architectures and hyperparameters
- **Hardware**: Use GPU acceleration for faster training

## 🐛 Troubleshooting

**Camera not working?**
- Check if camera permissions are granted
- Try changing camera index in `cv2.VideoCapture(0)` to `1` or `2`

**Low accuracy?**
- Ensure training data is properly labeled
- Increase training epochs
- Add more diverse training data

**Import errors?**
- Verify all required packages are installed
- Check Python version compatibility

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature')
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- TensorFlow/Keras team for the deep learning framework
- Facial expression datasets used for training

## 📞 Contact

saurav patel - patelsaurav1357@gmail.com

Project Link: [https://github.com/Saurav1357/facial_sentiment_analysis](https://github.com/Saurav1357/facial_sentiment_analysis)

---

⭐ **Star this repository if you found it helpful!**
