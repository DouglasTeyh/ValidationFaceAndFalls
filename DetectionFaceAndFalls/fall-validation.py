import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector

def treinar_modelo():
    try:
        modelo = cv2.face.LBPHFaceRecognizer_create()
        modelo.read('/home/douglas/Projects/FaceDetectionAndFalls/DetectionFaceAndFalls/classificadorLBPH.yml')
        print("\033[92m[INFO] Modelo de reconhecimento facial carregado com sucesso.\033[0m")
        return modelo
    except cv2.error as e:
        print(f"\033[91m[ERRO] Modelo não encontrado ou erro ao carregar: {e}\033[0m")
        return None

def detectar_e_coletar_quedas():
    detector = PoseDetector()
    ids_reais = []
    ids_previstos = []

    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        print("\033[91m[ERRO] Não foi possível acessar a câmera.\033[0m")
        return ids_reais, ids_previstos

    cam.set(3, 460)
    cam.set(4, 720)

    modelo_reconhecimento = treinar_modelo()

    if modelo_reconhecimento is None:
        print("\033[91m[ERRO] Não foi possível carregar o modelo de reconhecimento facial.\033[0m")
        return ids_reais, ids_previstos

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    dicionario_nomes = {
        1: "Douglas",
        2: "Daniel",
    }

    print("\033[92m[INFO] Use as teclas (1 para 'Queda', 2 para 'Sem Queda') ou ESC para sair.\033[0m")
    
    while True:
        ret, img = cam.read()
        if not ret:
            print("\033[91m[ERRO] Não foi possível capturar o frame da câmera.\033[0m")
            break

        img = detector.findPose(img, draw=False)
        pontos, bbox = detector.findPosition(img, draw=False)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rosto = gray[y:y + h, x:x + w]

            try:
                id_face, confianca = modelo_reconhecimento.predict(rosto)
                nome = dicionario_nomes.get(id_face, "Desconhecido")
                cv2.putText(img, nome, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            except cv2.error as e:
                print(f"\033[91m[ERRO] Erro ao prever rosto: {e}\033[0m")

        queda_detectada = False
        if pontos:
            cabeca = pontos[0][1]
            joelho = pontos[26][1] if len(pontos) > 26 else 0
            queda_detectada = joelho - cabeca <= 0

        queda_prevista_texto = "Queda Detectada" if queda_detectada else "Sem Queda"
        cor_queda = (0, 0, 255) if queda_detectada else (0, 255, 0)

        cv2.putText(img, queda_prevista_texto, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, cor_queda, 2)

        cv2.imshow('Queda e Reconhecimento Facial', img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  
            break
        elif key in [49, 50]: 
            ids_reais.append(1 if key == 49 else 0)
            ids_previstos.append(1 if queda_detectada else 0)
            print(f"\033[92m[INFO] ID real: {1 if key == 49 else 0}, Queda: {'Sim' if queda_detectada else 'Não'}\033[0m")

    cam.release()
    cv2.destroyAllWindows()

    return ids_reais, ids_previstos

if __name__ == "__main__":
    ids_reais, ids_previstos = detectar_e_coletar_quedas()
    print("\033[92m[INFO] Quedas detectadas.\033[0m")
