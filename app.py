'''
 Курсовая работа
 Цифровой ассистент для определения усталости

 Автор: Елизавета Розум 22303
 Дата окончания: 18.05.2023
'''

from deepface.detectors import FaceDetector
from deepface import DeepFace
import cv2
import numpy as np
import time

# --- считываем состояние
def parse_emotion(emotion):
    point = np.array(list(emotion.values())) / 100

    # -- emotions ----
    # -- angry, disgust, fear, happy, sad, surprise, neutral
    #  крайние значения шкалы
    top = np.array([0.235, 0.333, 0.078, 0.039, 0.49, 0.059, 0.353])
    bot = np.array([0.02, 0.02, 0, 0.68, 0.157, 0.078, 0.255])

    target = top - bot
    vector = point - bot

    # приведение векторов к стандартным
    target_norm = np.linalg.norm(target)
    #нормализуем вектор
    normalized_target = target / target_norm
    #находим проекцию на шкалу
    projection = np.dot(vector, normalized_target)
    #считаем процент усталости
    return np.linalg.norm(projection) / target_norm

def realtime_calc(source = 0):
    cam = cv2.VideoCapture(source)
    detector_name = 'opencv' # ssd - fast, dlib or mtcnn - slow, retinaface - very slow
    face_detector = FaceDetector.build_model(detector_name) #растягивает лицо, если вдруг оно к нам немного боком
    f_counter, e_frame = 0, 2 #задает, на какой кадр пересчитываем эмоции лиц
    face_locations = []
    emotions_hist = [] #история состояний(пара - временная метка и состояние)
    hest_depth = 2 * 3600 #2часовой ход
    crop = 1

    while True:
        ret, frame = cam.read() 
        frame = cv2.resize(frame, (0, 0), fx=1/1.5, fy=1/1.5)
        small_frame = cv2.resize(frame, (0, 0), fx=1/crop, fy=1/crop)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        if f_counter % e_frame == 0:
            face_locations, emotions = [], []
            for i, f in enumerate(FaceDetector.detect_faces(face_detector, detector_name, rgb_small_frame)):
                face_locations.append(f[1])
                c_face_img = f[0]
                e_pred = DeepFace.analyze(c_face_img, 
                                            actions=['emotion'], 
                                            enforce_detection=False, 
                                            detector_backend='opencv') #распределение вероятностей, какой шанс, что человек использует данную эмоцию
                state = parse_emotion(e_pred[0]['emotion'])

                timestamp = time.time()
                emotions_hist.append((timestamp, state))

            f_counter = 0
        f_counter += 1

        # отрисовываем формы
        for (x, y, w, h) in face_locations[:1]:
            color = (255, 255, 255)

            states, now = [], time.time()
            for t, s in emotions_hist[::-1]:
                if t < now - hest_depth: break
                else: states.append(s)
            curr_state = sum(states) / len(states)
            print(curr_state)

            cv2.rectangle(frame, 
                          (x * crop, y * crop), 
                          ((x + w) * crop, (y + h) * crop), 
                          color)
            cv2.rectangle(frame, 
                          (x * crop, (y + h) * crop), 
                          ((x + w) * crop, (y + h) * crop + 10), 
                          color, -1)
            cv2.rectangle(frame, 
                          (x * crop + 1, (y + h + 1) * crop), 
                          (int((x + w * curr_state) * crop) - 1, (y + h) * crop + 10 - 1), 
                          (275 - color[0], 255 - color[1], 255 - color[2]), -1)

        cv2.imshow('app', frame)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cv2.destroyAllWindows()


def main():
    realtime_calc("to_calc/exp.mov") # source can be "to_calc/<filename>"


if __name__ == '__main__':
    main()
