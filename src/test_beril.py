import cv2
import mediapipe as mp

# MediaPipe Yüz Ağı (Face Mesh) modelini başlat
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Bilgisayar kamerasını aç
cap = cv2.VideoCapture(0)
print("Kamera açılıyor... Kapatmak için açılan pencereye tıklayıp 'q' tuşuna basın.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü ayna gibi düzelt ve RGB formatına çevir
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Yüzdeki 468 noktayı tespit et
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            for lm in landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Beril - DSP Yuz Noktalari Testi (Cikis: q)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()