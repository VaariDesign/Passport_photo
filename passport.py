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


    height, width = image.shape[:2]

    # Scale the image to fit the screen
    if height > 800 or width > 800:
        scale_percent = 800/max(height, width)
        dimensions = (int(width * scale_percent), int(height * scale_percent))
        image = cv2.resize(image, dimensions, interpolation = cv2.INTER_AREA)


    x, y, h = 100, 100, 400
    # Finnish passport picture guide

    aspectratio = 36/47
    w = int(aspectratio * h)

    # help lines
    #
    head_top = int((4/47)*h)
    head_bot = int((6/47)*h)

    chin_top = int((38/47)*h)
    chin_bot = int((40/47)*h)

    nosel = int(w * (16.5/36))
    noser = int(w * (19.5/36))



    # Crop the image
    cv2.namedWindow("Image1")



    # Create trackbars to adjust the size and position of the cropping area
    cv2.createTrackbar("X", "Image1", x, image.shape[1], lambda x: None)
    cv2.createTrackbar("Y", "Image1", y, image.shape[0], lambda x: None)
    cv2.createTrackbar("H", "Image1", h, image.shape[0], lambda x: None)

    while True:
        # Get the current values of the trackbars
        x = cv2.getTrackbarPos("X", "Image1")
        y = cv2.getTrackbarPos("Y", "Image1")
        h = cv2.getTrackbarPos("H", "Image1")
        w = int(aspectratio * h)

        head_top_y = int((4/47)*h)
        head_bot_y = int((6/47)*h)

        chin_top_y = int((38/47)*h)
        chin_bot_y = int((40/47)*h)

        nosel_x = int(w * (16.5/36))
        noser_x = int(w * (19.5/36))


        # Draw a rectangle around the cropping area
        img_copy = image.copy()
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.line(img_copy, ((x+int(0.1*w)), y+ head_top_y), (x+int(0.9 * w), y+ head_top_y), (0, 0, 255), 1)
        cv2.line(img_copy, ((x+int(0.2*w)), y+ head_bot_y), (x+int(0.8 * w), y+ head_bot_y), (0, 0, 255), 1)

        cv2.line(img_copy, ((x+int(0.2*w)), y+ chin_top_y), (x+int(0.8 * w), y+ chin_top_y), (0, 0, 255), 1)
        cv2.line(img_copy, ((x+int(0.1*w)), y+ chin_bot_y), (x+int(0.9 * w), y+ chin_bot_y), (0, 0, 255), 1)

        cv2.line(img_copy, ((x+nosel_x), y+ int(0.4 * h)), ((x+nosel_x), y+ int(0.7 * h)), (0, 0, 255), 1)
        cv2.line(img_copy, ((x+noser_x), y + int(0.4 * h)), ((x+noser_x), y+ int(0.7 * h)), (0, 0, 255), 1)


        # Display the image with the cropping area
        cv2.imshow("Image1", img_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    imCrop = image[y:y+h, x:x+w]
    cv2.imwrite("output.jpg", imCrop)
    cv2.destroyAllWindows()

mode = "M" #input("Enter mode (A for Automatic / M for Manual): ")
if mode == "A":
    automatic_mode(image)
elif mode == "M":
    manual_mode(image)
else:
    print("Invalid mode selected.")

