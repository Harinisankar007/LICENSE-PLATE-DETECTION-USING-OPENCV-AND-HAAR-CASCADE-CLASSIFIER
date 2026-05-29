# LICENSE-PLATE-DETECTION-USING-OPENCV-AND-HAAR-CASCADE-CLASSIFIER

## Aim
To detect objects from an image using OpenCV and Haar Cascade Classifier techniques in Python.

## Software Required
- Anaconda – Python 3.7
- OpenCV
- NumPy
- Matplotlib
## Algorithm
Step 1: Import the necessary packages.

Step 2: Read and display the input image.

Step 3: Convert the image into grayscale.

Step 4: Apply Gaussian Blur and Histogram Equalization for preprocessing.

Step 5: Load the Haar Cascade Classifier XML file.

Step 6: Detect objects using detectMultiScale().

Step 7: Draw rectangles around detected objects.

Step 8: Save the detected object regions.

Step 9: Display the final detected output image.

## Program
## Developed By : HARINI S
## Register Number : 212224240049
```PYTHON
import cv2
import matplotlib.pyplot as plt
import os
import urllib.request

# -------------------------------------------------------------
# Step 1: Read and display the input image
# -------------------------------------------------------------
image_path = 'img.jpg'

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        "Image not found. Please check the image_path variable."
    )

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 2: Convert to grayscale
# -------------------------------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 3: Preprocessing
# -------------------------------------------------------------
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

equalized = cv2.equalizeHist(blurred)

plt.imshow(equalized, cmap='gray')
plt.title("Preprocessed Image")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 4: Load Haar Cascade
# -------------------------------------------------------------
cascade_path = 'haarcascade_frontalface_default.xml'

# Download cascade file automatically
if not os.path.exists(cascade_path):

    print("Cascade file not found. Downloading...")

    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

    urllib.request.urlretrieve(url, cascade_path)

    print("Cascade file downloaded successfully!")

# Load classifier
face_cascade = cv2.CascadeClassifier(cascade_path)

# -------------------------------------------------------------
# Step 5: Detect Faces
# -------------------------------------------------------------
faces = face_cascade.detectMultiScale(
    equalized,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print(f"Total Faces Detected: {len(faces)}")

# -------------------------------------------------------------
# Step 6: Draw Rectangles and Save Faces
# -------------------------------------------------------------
output = image.copy()

save_dir = "Detected_Faces"

os.makedirs(save_dir, exist_ok=True)

for i, (x, y, w, h) in enumerate(faces):

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3
    )

    face_crop = image[y:y+h, x:x+w]

    save_path = f"{save_dir}/face_{i+1}.jpg"

    cv2.imwrite(save_path, face_crop)

if len(faces) > 0:
    print(f"{len(faces)} face(s) saved in '{save_dir}' folder.")
else:
    print("No faces detected.")

# -------------------------------------------------------------
# Step 7: Display Final Output
# -------------------------------------------------------------
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Faces")
plt.axis('off')
plt.show()
```
## Output
## Original Image
<img width="489" height="289" alt="image" src="https://github.com/user-attachments/assets/cf9a70de-075d-4f88-bc43-30d6bbbb2b08" />

## Grayscale Image
<img width="458" height="302" alt="image" src="https://github.com/user-attachments/assets/e28b15f3-4475-49d5-8881-11927166d84c" />

## Preprocessed Image
<img width="489" height="358" alt="image" src="https://github.com/user-attachments/assets/61ac84f4-8b6e-4d31-8cf4-bed66af8e7b4" />

## Detected Output Image
<img width="498" height="292" alt="image" src="https://github.com/user-attachments/assets/c74cf89a-0989-4be9-96d2-f6515bcd9536" />

## Result
Thus the object detection using OpenCV and Haar Cascade Classifier was successfully implemented in Python, and the detected objects were displayed and saved successfully.

