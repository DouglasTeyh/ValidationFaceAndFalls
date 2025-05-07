import cv2
import numpy as np

def reconhecer_e_coletar_dados():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('/home/douglas/Projects/FaceDetectionAndFalls/DetectionFaceAndFalls/classificadorLBPH.yml')
    faceCascade = cv2.CascadeClassifier("/home/douglas/Projects/FaceDetectionAndFalls/DetectionFaceAndFalls/haarcascade_frontalface_default.xml")
    names = ['Desconhecido', 'Douglas', 'Daniel']  # IDs e nomes cadastrados no sistema

    ids_reais = []  # IDs reais
    ids_previstos = []  # IDs previstos

    cam = cv2.VideoCapture(1)
    cam.set(3, 460)
    cam.set(4, 720)

    print("[INFO] Pressione (0-9) para ID real ou 'ESC' para sair.")
    
    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERRO] Câmera não disponível.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            id_previsto, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            confidence_percentage = 100 - confidence

            if confidence_percentage < 30:
                name = "Desconhecido"
                id_previsto = 0
            elif id_previsto < len(names):
                name = names[id_previsto]
            else:
                name = "Desconhecido"
                id_previsto = 0

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow('Reconhecimento', img)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
        elif 48 <= key <= 57:
            id_real = key - 48
            ids_reais.append(id_real)
            ids_previstos.append(id_previsto)
            print(f"[INFO] ID real: {id_real}, ID previsto: {id_previsto}")

        if len(ids_reais) >= 100:
            break

    cam.release()
    cv2.destroyAllWindows()

    ids_reais = [id for id in ids_reais if id != 0]
    ids_previstos = [id for id in ids_previstos if id != 0]

    return ids_reais, ids_previstos

if __name__ == "__main__":
    ids_reais, ids_previstos = reconhecer_e_coletar_dados()
    if ids_reais and ids_previstos:
        print("[INFO] Dados de validação coletados com sucesso.")
    else:
        print("[INFO] Nenhum dado coletado.")
