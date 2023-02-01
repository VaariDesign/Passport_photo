import cv2
import numpy as np
import mediapipe as mp


print(cv2. __version__)
# Load the cascade classifier for detecting faces
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

# Read the image
image = cv2.imread("input.jpg")


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def autorotate_image(img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("No faces found")
    else:
        for (x, y, w, h) in faces:
            print(faces)
            face_center = (x + w//2, y + h//2)
            face = gray_img[y:y+h, x:x+w]
            # Draw a rectangle around the face
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

            eyes = eye_cascade.detectMultiScale(gray_img[y:y+h, x:x+w], scaleFactor=1.1, minNeighbors=5)
            if len(eyes) < 2:
                print("Couldn't detect both eyes")
            else:
                for ex, ey, ew, eh in eyes:
                    cv2.rectangle(img[y:y+h, x:x+w], (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                left_eye, right_eye = eyes[:2]
                left_eye = (left_eye[0] + x, left_eye[1] + y)
                right_eye = (right_eye[0] + x, right_eye[1] + y)
                nose_x, nose_y = (left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2
                angle = np.degrees(np.arctan2(nose_y - img.shape[0]//2, nose_x - img.shape[1]//2))
                print(left_eye)
                print(right_eye)
                print(angle)
                print(nose_x)
                print(nose_y)


                cv2.imshow("Rotated Image", img)
                cv2.waitKey(0)


                M = cv2.getRotationMatrix2D(face_center, angle, 1)
                rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    cv2.imshow("Rotated Image", rotated_img)
    cv2.waitKey(0)
    return rotated_img

def locate_face(image):
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, static_image_mode=True, refine_landmarks=True)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:

        for faceLms in results.multi_face_landmarks:
            print(faceLms)
            mpDraw.draw_landmarks(image, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec)

    return image



def manual_mode(image):
    # Get the dimensions of the imageM


    height, width = image.shape[:2]

    # Scale the image to fit the screen
    if height > 1000 or width > 1000:
        scale_percent = 1000/max(height, width)
        dimensions = (int(width * scale_percent), int(height * scale_percent))
        image = cv2.resize(image, dimensions, interpolation = cv2.INTER_AREA)


    x, y, h = 100, 100, 400
    rotate = 0

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
    cv2.namedWindow("Image12")

    #Rotate image
    #image = autorotate_image(image)

    cv2.imshow("Image1", autorotate_image(image))


    # Create trackbars to adjust the size and position of the cropping area
    cv2.createTrackbar("X", "Image12", x, image.shape[1], lambda x: None)
    cv2.createTrackbar("Y", "Image12", y, image.shape[0], lambda x: None)
    cv2.createTrackbar("H", "Image12", h, image.shape[0], lambda x: None)
    cv2.createTrackbar("Rotate", "Image12", rotate, 360, lambda x: None)

    while True:
        # Get the current values of the trackbars
        x = cv2.getTrackbarPos("X", "Image12")
        y = cv2.getTrackbarPos("Y", "Image12")
        h = cv2.getTrackbarPos("H", "Image12")
        rotate = cv2.getTrackbarPos("Rotate", "Image12")
        w = int(aspectratio * h)




        head_top_y = int((4/47)*h)
        head_bot_y = int((6/47)*h)
        avg_head = int((head_top_y + head_bot_y)/2)

        chin_top_y = int((38/47)*h)
        chin_bot_y = int((40/47)*h)
        avg_chin = int((chin_bot_y+chin_top_y)/2)

        nosel_x = int(w * (16.5/36))
        noser_x = int(w * (19.5/36))


        # Draw a rectangle around the cropping area
        img_copy = image.copy()
        #img_copy = locate_face(img_copy)
        img_copy = rotate_image(img_copy, rotate)
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 1)

        #Draws Help lines
        cv2.line(img_copy, ((x+int(0.1*w)), y+ head_top_y), (x+int(0.9 * w), y+ head_top_y), (0, 0, 255), 1)
        cv2.line(img_copy, ((x+int(0.2*w)), y+ head_bot_y), (x+int(0.8 * w), y+ head_bot_y), (0, 0, 255), 1)

        cv2.line(img_copy, ((x+int(0.2*w)), y+ chin_top_y), (x+int(0.8 * w), y+ chin_top_y), (0, 0, 255), 1)
        cv2.line(img_copy, ((x+int(0.1*w)), y+ chin_bot_y), (x+int(0.9 * w), y+ chin_bot_y), (0, 0, 255), 1)

        cv2.line(img_copy, ((x+nosel_x), y+ int(0.4 * h)), ((x+nosel_x), y+ int(0.7 * h)), (0, 0, 255), 1)
        cv2.line(img_copy, ((x+noser_x), y + int(0.4 * h)), ((x+noser_x), y+ int(0.7 * h)), (0, 0, 255), 1)

        cv2.ellipse(img_copy, ((x+int(w/2)),y+int((avg_head+avg_chin)/2)), (int(0.3 * w),(int((avg_chin-avg_head)/2))),0, 0, 360, (0, 0, 255), 1)




        # Display the image with the cropping area
        cv2.imshow("Image1", img_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    image = rotate_image(image, rotate)
    imCrop = image[y:y+h, x:x+w]
    cv2.imwrite("output.jpg", imCrop)
    cv2.destroyAllWindows()

mode = "M" #input("Enter mode (A for Automatic / M for Manual): ")
if mode == "A":
    print("nope")
elif mode == "M":
    manual_mode(image)
else:
    print("Invalid mode selected.")

