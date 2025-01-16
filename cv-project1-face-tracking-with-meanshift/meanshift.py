# Title: Meanshift Tracking

# This code implements face tracking using the Meanshift algorithm in OpenCV. 
# It starts by loading a video file and setting up a video writer for the output. 
# The program first detects a face in the initial frame using a Haar Cascade Classifier, 
# then establishes a Region of Interest (ROI) around that face. 
# It converts the ROI to HSV color space and calculates a histogram of the hue channel 
# to create a model of the face's color distribution. 
# The main loop then processes each frame of the video, converting it to HSV and 
# using calcBackProject to create a probability map of where the face might be based on the histogram. 
# The Meanshift algorithm (cv2.meanShift) is applied to this probability map to track the face's movement, 
# updating the tracking window position in each frame. 
# The tracked face is visualized with a rectangle, and the processed frames are written to an output video file. 
# The program continues until either the video ends or the user presses the ESC key.

import cv2

input_video_path = 'input_videos/visionface.mp4'
output_video_path = f'{input_video_path.split(".")[0]}_meanshift_output.avi'

# Read the video
cap = cv2.VideoCapture(input_video_path)

# Get fps, width and height of video stream
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Take the first frame and detect face
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Initialize tracking window from the first detected face
if len(faces) > 0:
    x, y, w, h = faces[0]  # Use the first detected face
    track_window = (x, y, w, h)
else:
    print("No face detected in the first frame")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iterations or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, 
                     fourcc,
                     fps, 
                     (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on the image
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        
        # Write the frame to output video
        out.write(img2)
        
        cv2.imshow('Tracking', img2)

        if cv2.waitKey(30) & 0xFF == 27:
            break
    else:
        break

# Release everything
cap.release()
out.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
