# Video Activity Recognition using Conv-LSTM (Convolutional Long Short-Term Memory)


Why Conv-LSTM?
Conv-LSTM (Convolutional Long Short-Term Memory) is an advanced deep learning architecture designed for spatiotemporal sequence processing. It effectively captures both spatial features (using convolution layers) and temporal dependencies (using LSTM units), making it well-suited for video-based activity recognition.

Key Benefits of Using Conv-LSTM for Video Activity Recognition:
Capturing Spatial Features: CNN layers extract meaningful spatial features from individual frames.
Understanding Temporal Dynamics: LSTM layers analyze sequential dependencies across frames, making it ideal for tracking activities over time.
Better than Frame-Wise CNNs: Unlike simple CNN-based classifiers that process each frame independently, Conv-LSTM maintains context over time, improving recognition accuracy.
Handling Varying Video Lengths: LSTM’s sequential processing allows flexible input lengths, accommodating videos of different durations.


Dataset & Preprocessing:
The dataset, stored on Google Drive, consists of video sequences that are preprocessed by extracting frames, resizing them to 64×64 pixels, normalizing pixel values, and feeding them into the model. Training was conducted on Google Colab using various Python libraries, including:

TensorFlow for deep learning
OpenCV & MoviePy for video processing
Matplotlib for visualization
Scikit-learn for dataset splitting
ConvLSTM Model Architecture:
The ConvLSTM-based model follows a robust deep learning pipeline consisting of:

Two ConvLSTM2D layers with tanh activation
MaxPooling3D layers for spatial downsampling
Batch Normalization for training stability
Dropout layers to prevent overfitting
GlobalAveragePooling3D to aggregate spatiotemporal features
Fully connected dense layer (64 units, ReLU activation)
Softmax layer for multi-class classification
Training & Optimization:
The model is trained using the Adam optimizer with a learning rate of 0.0001, categorical crossentropy loss, and accuracy as the evaluation metric.

Training Configuration:
150 epochs with batch size = 32
EarlyStopping (stops training if validation loss doesn’t improve for 90 epochs)
ReduceLROnPlateau (reduces learning rate if validation loss doesn’t improve for 25 epochs)
Data Split: 75% for training, 25% for testing
Performance & Evaluation:
After extensive training and testing, the ConvLSTM model achieves an accuracy of 80.95%, demonstrating its effectiveness in recognizing human activities in video sequences.
