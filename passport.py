import cv2

# Load the cascade classifier for detecting faces
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Read the image
image = cv2.imread("input.jpg")


def automatic_mode(image):
    # Detect faces in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # If a face is detected
    if len(faces) > 0:
        # Get the coordinates of the face
        x, y, w, h = faces[0]
        # Crop the image to the size of the face
        image = image[y:y+h, x:x+w]
        # Save the image
        cv2.imwrite("output.jpg", image)
    else:
        print("No face detected in the image.")

def manual_mode(image):
    # Get the dimensions of the imageM

    height, width = image.shape[:2]

    # Scale the image to fit the screen
    if height > 1200 or width > 1200:
        scale_percent = 1200/max(height, width)
        dimensions = (int(width * scale_percent), int(height * scale_percent))
        image = cv2.resize(image, dimensions, interpolation = cv2.INTER_AREA)


    r = cv2.selectROI(image)
    imCrop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv2.imwrite("output.jpg", imCrop)
    cv2.destroyAllWindows()

mode = input("Enter mode (A for Automatic / M for Manual): ")
if mode == "A":
    automatic_mode(image)
elif mode == "M":
    manual_mode(image)
else:
    print("Invalid mode selected.")
