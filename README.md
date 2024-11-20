# 💻 Computer Vision: Course

This repository houses a collection assignments from the 3rd semester computer vision course of my Master's in Photonics for Security, Reliability, and Safety (PSRS) at UPEC University, Paris Est-Créteil

## 🎥 1. Meanshift Face Tracking

This project implements face tracking using the Meanshift algorithm in OpenCV. 

**How it Works:**

1. **Face Detection:** The program starts by detecting a face in the first frame of the video using a Haar Cascade Classifier. 
2. **ROI Selection:** A Region of Interest (ROI) is established around the detected face.
3. **Color Model:** The ROI is converted to HSV color space, and a histogram of the hue channel is calculated to create a model of the face's color distribution.
4. **Probability Map:** Each frame is converted to HSV, and `calcBackProject` is used to create a probability map of where the face might be based on the histogram.
5. **Meanshift Tracking:** The Meanshift algorithm (`cv2.meanShift`) is applied to this probability map to track the face's movement, updating the tracking window position in each frame.
6. **Visualization:** The tracked face is visualized with a rectangle, and the processed frames are written to an output video file.

**Code Structure:**

- **`meanshift.py`:** Contains the main implementation of the Meanshift face tracking algorithm.

**Example Output:**

The output video will show the original video with a rectangle tracking the face.

## 🖼️ 2. GoogLeNet on Labeled Faces in the Wild (LFW) Dataset

This project implements face recognition using the GoogLeNet architecture, tested on the Labeled Faces in the Wild (LFW) dataset.

**How it Works:**

1. **Dataset Preparation:** The LFW dataset is preprocessed and deep funneled for consistent alignment of facial features.
2. **Model Architecture:** GoogLeNet, a convolutional neural network designed for image classification tasks, is employed for this project.
3. **Training and Testing:** The network is trained on a subset of the LFW dataset and validated on unseen data. GPU acceleration is utilized for faster training.
4. **Performance Evaluation:** Model performance is measured using accuracy and confusion matrices on the test dataset.
5. **Implementation:** Includes setup for GPU usage, model training, and visualization of results.

**Code Structure:**

- **`cv_assignment2_googlenet.ipynb`:** Contains the full implementation, including data preprocessing, model training, and evaluation.

**Example Output:**

The project demonstrates the classification performance of GoogLeNet on recognizing faces in the LFW dataset, showcasing its capability in high-dimensional feature extraction.


## 📂 Repository Structure
- Each session's project is stored in its respective folder with the code and resources.
- Feel free to explore the individual assignments! 😎
