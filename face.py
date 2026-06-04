import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Webcam start
video = cv2.VideoCapture(0)

# Drawing styles
drawing_spec = mp_drawing.DrawingSpec(
    color=(0, 255, 255),   # Yellow
    thickness=1,
    circle_radius=1
)

contour_spec = mp_drawing.DrawingSpec(
    color=(255, 0, 255),   # Pink
    thickness=2
)

# Face Mesh setup
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = video.read()

        if not ret:
            print("Camera not working")
            break

        # Flip image for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image
        results = face_mesh.process(image)

        # Convert back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                # Mesh
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec
                )

                # Contours
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=contour_spec
                )

        # Show output
        cv2.imshow("Face Mesh", image)

        # Exit on pressing q
        if cv2.waitKey(1) == ord('q'):
            break

# Release resources
video.release()
cv2.destroyAllWindows()