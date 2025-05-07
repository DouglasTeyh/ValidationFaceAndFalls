import cv2
import os
import time

def verde(msg): return f"\033[92m{msg}\033[0m"
def amarelo(msg): return f"\033[93m{msg}\033[0m"
def vermelho(msg): return f"\033[91m{msg}\033[0m"

cam = cv2.VideoCapture(1)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
os.makedirs("dataset", exist_ok=True)

face_id = input("\nDigite o ID do usuário: ")
qtdImgs = int(input("\nDigite a quantidade de imagens a serem capturadas: "))
count = 0

print(verde("\n[INFO] Iniciando a captura de faces..."))
print(verde("[INFO] Pressione ESC para parar.\n"))

while count < qtdImgs:
    ret, img = cam.read()
    if not ret:
        print(vermelho("[ERRO] Falha ao capturar imagem da câmera."))
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print(amarelo("[ALERT] Nenhuma face detectada."))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_image = gray[y:y + h, x:x + w]
        cv2.imwrite(f"dataset/User.{face_id}.{count + 1}.jpg", face_image)
        count += 1
        print(verde(f"[INFO] Imagem {count} capturada."))
        time.sleep(0.02)

    cv2.imshow('Face Capture', img)
    if cv2.waitKey(100) & 0xFF == 27:
        print(verde("\n[INFO] Captura interrompida pelo usuário."))
        break

print(verde("\n[INFO] Captura encerrada."))
cam.release()
cv2.destroyAllWindows()
