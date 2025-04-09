import os
import cv2
import mediapipe as mp

def extrair_dados_video(video_path, rotulo):
    """
    Extrai os keypoints de um vídeo com MediaPipe Pose e atribui um rótulo (1 ou 0).
    Retorna uma lista de frames com os keypoints.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    cap = cv2.VideoCapture(video_path)
    dados = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = pose.process(imagem_rgb)

        if resultados.pose_landmarks:
            frame_data = []
            for lm in resultados.pose_landmarks.landmark:
                frame_data.extend([lm.x, lm.y, lm.z, lm.visibility])
            frame_data.append(rotulo)
            dados.append(frame_data)

    cap.release()
    return dados
