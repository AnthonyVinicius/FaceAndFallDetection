import face_recognition
import cv2
import os
import mediapipe as mp

# === CONFIGURAÇÕES ===
FACESDIR = "Data"
THRESHOLD_RECONHECIMENTO = 0.6  # Sensibilidade do reconhecimento facial

# === INICIALIZAÇÕES ===
dataEncodings = []
dataNames = []

# === Carrega imagens e codifica ===
for filename in os.listdir(FACESDIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(FACESDIR, filename)
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)

        if encoding:
            dataEncodings.append(encoding[0])
            dataNames.append(os.path.splitext(filename)[0])
        else:
            print(f"[!] Nenhum rosto detectado em {filename}, ignorado.")

if not dataEncodings:
    print("Erro: Nenhum rosto foi carregado. Verifique a pasta 'Data'.")
    exit()

# === Inicializa webcam ===
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not video_capture.isOpened():
    print("Erro: Não foi possível acessar a webcam.")
    exit()

# === Inicializa MediaPipe ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === Função simples para detectar queda ===
def queda_detectada(results):
    try:
        shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        if shoulder.y > 0.8:  # Ombro muito próximo do "chão" (parte inferior da imagem)
            return True
        return False
    except:
        return False

# === Loop principal ===
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Reduz a imagem para acelerar o reconhecimento facial
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # === Reconhecimento facial ===
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(dataEncodings, face_encoding)

        name = "Pessoa não cadastrada"
        if len(distances) > 0 and distances.min() < THRESHOLD_RECONHECIMENTO:
            best_match_index = distances.argmin()
            name = dataNames[best_match_index]

        # Redimensiona de volta para o tamanho original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenha caixa e nome
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 5, bottom - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # === Detecção de pose e queda ===
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if queda_detectada(results):
            cv2.putText(frame, "QUEDA DETECTADA", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # === Mostrar o resultado ===
    cv2.imshow("Reconhecimento Facial + Detecção de Queda", frame)

    # Tecla ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

# === Finalização ===
video_capture.release()
cv2.destroyAllWindows()
