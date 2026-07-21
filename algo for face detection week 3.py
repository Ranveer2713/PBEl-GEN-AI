import cv2
import os

# Load Haar Cascade
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if detector.empty():
    print("Error loading Haar Cascade classifier")
    exit()

# Create main dataset folder
if not os.path.exists("DataSet1"):
    os.mkdir("DataSet1")

# User input
name = input("Enter person's name: ")
num_samples = int(input("Enter number of images to capture: "))

folder = os.path.join("DataSet1", name.lower())

if not os.path.exists(folder):
    os.mkdir(folder)
else:
    print("Folder already exists. Images will be added.")

# Open webcam
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Could not open camera")
    exit()

sample = 0

print("\nPress 'c' to start capturing images.")
print("Press 'q' to quit.\n")

capturing = False

while True:
    ret, frame = cam.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Save images only after pressing C
        if capturing and sample < num_samples:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (80, 100))

            sample += 1

            image_path = os.path.join(
                folder,
                f"{sample}.jpg"
            )

            cv2.imwrite(image_path, face_img)

            print(f"Saved image {sample}/{num_samples}")

    cv2.imshow("Face Dataset Creator", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        capturing = True
        print("Capturing started...")

    if key == ord('q'):
        break

    if sample >= num_samples:
        print("Dataset creation completed.")
        break

cam.release()
cv2.destroyAllWindows()