import cv2
import mediapipe as mp
import time
import numpy as np


class FaceMeshDetector:

    def __init__(self, static_image_mode: bool = False, max_num_faces: int = 1, min_detection_confidence: bool = 0.5,
                 min_tracking_confidence: float = 0.5, thickness: int = 1, circle_radius: int = 1) -> None:
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.thickness = thickness
        self.circle_radius = circle_radius

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode, self.max_num_faces,
                                                 self.min_detection_confidence, self.min_tracking_confidence)
        self.drawSpecs = self.mpDraw.DrawingSpec(self.thickness, self.circle_radius)

    def findFaceMesh(self, img: np.ndarray, draw: bool = True) -> [np.ndarray, list[list[list[int]]]]:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLandmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLandmarks, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpecs, self.drawSpecs)
                face = []
                for id, landmark in enumerate(faceLandmarks.landmark):
                    imageHeight, imageWidth, imageChannel = img.shape
                    x, y = int(landmark.x * imageWidth), int(landmark.y * imageHeight)
                    face.append([x, y])
                faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0

    meshDetector = FaceMeshDetector()

    while True:
        success, img = cap.read()

        img, faces = meshDetector.findFaceMesh(img)

        if len(faces):
            print(len(faces))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break


if __name__ == "__main__":
    main()

