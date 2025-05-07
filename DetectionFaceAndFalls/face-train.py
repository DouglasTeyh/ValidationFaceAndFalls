import cv2
import numpy as np
import os

def verde(msg): return f"\033[92m{msg}\033[0m"
def vermelho(msg): return f"\033[91m{msg}\033[0m"
def amarelo(msg): return f"\033[93m{msg}\033[0m"

caminho_treinamento = "/home/douglas/Projects/FaceDetectionAndFalls/dataset"
treinamento_dados, ids = [], []
detector_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for imagem_nome in os.listdir(caminho_treinamento):
    caminho_imagem = os.path.join(caminho_treinamento, imagem_nome)

    if not os.path.isfile(caminho_imagem):
        print(vermelho(f"[ERRO] Arquivo não encontrado ou não é uma imagem: {imagem_nome}"))
        continue

    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        print(vermelho(f"[ERRO] Imagem não encontrada ou inválida: {imagem_nome}"))
        continue

    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faces = detector_face.detectMultiScale(imagem_cinza, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(amarelo(f"[ALERT] Nenhuma face detectada em {imagem_nome}. Imagem ignorada."))
        continue

    try:
        partes_nome = imagem_nome.split('.')
        id_usuario = int(partes_nome[1])
    except ValueError:
        print(vermelho(f"[ERRO] Nome de arquivo inválido (não segue formato User.ID.count.jpg): {imagem_nome}"))
        continue

    for (x, y, l, a) in faces:
        rosto = imagem_cinza[y:y+a, x:x+l]
        treinamento_dados.append(rosto)
        ids.append(id_usuario)

ids_np = np.array(ids)
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.train(treinamento_dados, ids_np)
reconhecedor.save('DetectionFaceAndFalls/classificadorLBPH.yml')

print(verde("[INFO] Treinamento finalizado com sucesso!"))
