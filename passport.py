import cv2

print(cv2. __version__)
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
    aspect_ratio = 4/3

    height, width = image.shape[:2]

    # Scale the image to fit the screen
    if height > 1200 or width > 1200:
        scale_percent = 1200/max(height, width)
        dimensions = (int(width * scale_percent), int(height * scale_percent))
        image = cv2.resize(image, dimensions, interpolation = cv2.INTER_AREA)

    # Crop the image
    cv2.namedWindow("Image1")
    x, y, w, h = 100, 100, 200, 200

    # Create trackbars to adjust the size and position of the cropping area
    cv2.createTrackbar("X", "Image1", x, image.shape[1], lambda x: None)
    cv2.createTrackbar("Y", "Image1", y, image.shape[0], lambda x: None)
    cv2.createTrackbar("W", "Image1", w, image.shape[1], lambda x: None)
    cv2.createTrackbar("H", "Image1", h, image.shape[0], lambda x: None)

    while True:
        # Get the current values of the trackbars
        x = cv2.getTrackbarPos("X", "Image1")
        y = cv2.getTrackbarPos("Y", "Image1")
        w = cv2.getTrackbarPos("W", "Image1")
        h = cv2.getTrackbarPos("H", "Image1")

        # Draw a rectangle around the cropping area
        img_copy = image.copy()
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the image with the cropping area
        cv2.imshow("Image", img_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    imCrop = image[y:y+h, x:x+w]
    cv2.imwrite("output.jpg", imCrop)
    cv2.destroyAllWindows()

mode = input("Enter mode (A for Automatic / M for Manual): ")
if mode == "A":
    automatic_mode(image)
elif mode == "M":
    manual_mode(image)
else:
    print("Invalid mode selected.")
