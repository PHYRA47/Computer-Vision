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

## 📂 Repository Structure
- Each session's project is stored in its respective folder with the code and resources.
- Feel free to explore the individual assignments! 😎
