import os
import re
import cv2
import whisper
import argparse
import logging
import threading
import glob
import warnings
from tqdm import tqdm
from time import time
from PIL import Image
import pytesseract
from langdetect import detect, DetectorFactory
from spellchecker import SpellChecker
from queue import Queue
import numpy as np

# Suprimir avisos específicos
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# Fixar a seed para resultados consistentes na detecção de idioma
DetectorFactory.seed = 0

# Inicializar o corretor ortográfico para português
corretor = SpellChecker(language="pt")

# Configurar o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def texto_legivel(texto, min_palavra=4):
    """
    Verifica se o texto contém pelo menos uma palavra legível com o tamanho mínimo especificado.
    """
    try:
        # Extrair palavras com letras A-Z, a-z, 0-9
        palavras = re.findall(r'\b[A-Za-z0-9]{' + str(min_palavra) + r',}\b', texto)
        if not palavras:
            return False

        # Verificar ortografia das palavras
        palavras_legiveis = [palavra for palavra in palavras if palavra.lower() in corretor]

        if palavras_legiveis:
            # Unir as palavras legíveis para detecção de idioma
            texto_para_deteccao = ' '.join(palavras_legiveis)
            # Detectar idioma
            idioma = detect(texto_para_deteccao)
            if idioma == 'pt':
                return True

        return False
    except Exception as e:
        logging.error(f"Erro na validação do texto: {e}")
        return False

def formatar_timestamp_para_nome(timestamp_ms):
    """Formata o timestamp em milissegundos para o formato HH_MM_SS.FFFF."""
    try:
        total_segundos = timestamp_ms / 1000.0
        horas = int(total_segundos // 3600)
        minutos = int((total_segundos % 3600) // 60)
        segundos = int(total_segundos % 60)
        milissegundos = int((total_segundos - int(total_segundos)) * 10000)
        return f"{horas:02d}_{minutos:02d}_{segundos:02d}.{milissegundos:04d}"
    except Exception as e:
        logging.error(f"Erro ao formatar timestamp para nome: {e}")
        return "00_00_00.0000"

def detectar_texto(frame, net, min_confidence=0.5):
    """
    Utiliza o EAST Text Detector para detectar regiões com texto no frame.
    Retorna uma lista de caixas delimitadoras das regiões de texto.
    """
    try:
        orig = frame.copy()
        (H, W) = frame.shape[:2]

        # Definir o tamanho para o modelo (deve ser múltiplo de 32)
        (newW, newH) = (320, 320)
        rW = W / float(newW)
        rH = H / float(newH)

        # Redimensionar a imagem
        frame_resized = cv2.resize(frame, (newW, newH))
        (H, W) = frame_resized.shape[:2]

        # Definir a imagem como blob para o modelo
        blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        # Obter as pontuações e geometria
        (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid",
                                          "feature_fusion/concat_3"])

        # Decodificar as predições
        (rects, confidences) = decode_predictions(scores, geometry, min_confidence)

        # Aplicar Non-Maxima Suppression para suprimir caixas sobrepostas
        indices = cv2.dnn.NMSBoxesRotated(rects, confidences, min_confidence, 0.4)

        result_boxes = []
        if len(indices) > 0:
            for i in indices:
                # Obter as coordenadas da caixa
                vertices = cv2.boxPoints(rects[i[0]])
                # Ajustar as coordenadas para o tamanho original da imagem
                vertices[:, 0] *= rW
                vertices[:, 1] *= rH
                # Converter para inteiros
                vertices = np.int0(vertices)
                result_boxes.append(vertices)
        return result_boxes
    except Exception as e:
        logging.error(f"Erro na detecção de texto: {e}")
        return []

def decode_predictions(scores, geometry, min_confidence):
    """
    Decodifica as predições do EAST Text Detector.
    """
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        # Extrair pontuações e geometria
        scoresData = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            score = scoresData[x]

            if score < min_confidence:
                continue

            # Calcular offset
            offsetX = x * 4.0
            offsetY = y * 4.0

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x0[x] + x2[x]
            w = x1[x] + x3[x]

            # Calcular centro
            endX = int(offsetX + cos * x1[x] + sin * x2[x])
            endY = int(offsetY - sin * x1[x] + cos * x2[x])
            startX = int(endX - w)
            startY = int(endY - h)
            centerX = (startX + endX) / 2.0
            centerY = (startY + endY) / 2.0

            rect = ((centerX, centerY), (w, h), -1 * angle * 180.0 / np.pi)
            rects.append(rect)
            confidences.append(float(score))

    return (rects, confidences)

def processar_frames_com_opencv(caminho_video, pasta_saida, fps_extracao=4, min_palavra=4):
    """Processa os frames do vídeo usando OpenCV e salva apenas aqueles com texto detectado."""
    try:
        # Carregar o modelo EAST
        net = cv2.dnn.readNet('frozen_east_text_detection.pb')

        cap = cv2.VideoCapture(caminho_video)

        if not cap.isOpened():
            logging.error(f"Não foi possível abrir o vídeo: {caminho_video}")
            return

        fps_video = cap.get(cv2.CAP_PROP_FPS)
        frame_intervalo = int(fps_video / fps_extracao) if fps_extracao else 1

        frame_num = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Processando frames", unit="frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Obter o timestamp do frame em milissegundos
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Processar apenas a cada 'frame_intervalo' frames
            if frame_num % frame_intervalo == 0:
                try:
                    # Detectar texto no frame
                    caixas = detectar_texto(frame, net)

                    if caixas:
                        # Se houver caixas detectadas, extrair as regiões e aplicar OCR
                        texto_total = ""
                        for vertices in caixas:
                            # Obter a caixa delimitadora dos vértices
                            x, y, w, h = cv2.boundingRect(vertices)
                            # Garantir que as coordenadas estejam dentro dos limites da imagem
                            x = max(0, x)
                            y = max(0, y)
                            w = min(frame.shape[1] - x, w)
                            h = min(frame.shape[0] - y, h)

                            # Extrair a região de interesse (ROI)
                            roi = frame[y:y+h, x:x+w]
                            # Converter para PIL Image
                            imagem_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                            # Executar OCR na ROI
                            configuracao_tesseract = (
                                r'--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -l por'
                            )
                            texto = pytesseract.image_to_string(imagem_pil, config=configuracao_tesseract)
                            texto_total += " " + texto

                        if texto_total.strip() and texto_legivel(texto_total, min_palavra):
                            # Formatar o timestamp em HH_MM_SS.FFFF
                            timestamp_formatado = formatar_timestamp_para_nome(timestamp_ms)
                            # Salvar o frame completo com texto
                            nome_frame = f"frame_{timestamp_formatado}.png"
                            caminho_frame_saida = os.path.join(pasta_saida, nome_frame)
                            cv2.imwrite(caminho_frame_saida, frame)

                            # Salvar o texto extraído
                            caminho_texto_saida = os.path.join(pasta_saida, f"frame_{timestamp_formatado}.txt")
                            with open(caminho_texto_saida, "w", encoding="utf-8") as arquivo_texto:
                                arquivo_texto.write(texto_total.strip())
                except Exception as e:
                    logging.warning(f"Erro ao processar um frame: {e}")
                    continue

            frame_num += 1
            pbar.update(1)

        pbar.close()
        cap.release()
    except Exception as e:
        logging.error(f"Erro ao processar frames com OpenCV: {e}")
        raise

def transcrever_audio(caminho_video, nome_modelo="medium", idioma="Portuguese"):
    """Transcreve o áudio do vídeo usando o Whisper."""
    try:
        # Suprimir avisos do Whisper
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo_whisper = whisper.load_model(nome_modelo)
            resultado = modelo_whisper.transcribe(caminho_video, language=idioma, task="transcribe")

        caminho_srt = caminho_video.replace(".mp4", ".srt")
        caminho_fala_cronometrada = caminho_video.replace(".mp4", "-Fala.Cronometrada.txt")

        with open(caminho_srt, "w", encoding="utf-8") as arquivo_srt:
            for segmento in resultado['segments']:
                inicio = segmento['start']
                fim = segmento['end']
                texto = segmento['text'].strip()

                arquivo_srt.write(f"{segmento['id']}\n")
                arquivo_srt.write(f"{formatar_timestamp(inicio)} --> {formatar_timestamp(fim)}\n")
                arquivo_srt.write(f"{texto}\n\n")

        with open(caminho_fala_cronometrada, "w", encoding="utf-8") as arquivo_txt:
            for segmento in resultado['segments']:
                inicio = formatar_timestamp(segmento['start'])
                texto = segmento['text'].strip()
                arquivo_txt.write(f"{inicio}: {texto}\n")

        logging.info(f"Arquivos de transcrição gerados: {caminho_srt}, {caminho_fala_cronometrada}")
    except Exception as e:
        logging.error(f"Erro ao transcrever áudio para {caminho_video}: {e}")
        raise

def formatar_timestamp(segundos):
    """Formata segundos no formato de timestamp para SRT."""
    try:
        horas = int(segundos // 3600)
        minutos = int((segundos % 3600) // 60)
        segs = int(segundos % 60)
        milissegundos = int((segundos - int(segundos)) * 1000)
        return f"{horas:02d}:{minutos:02d}:{segs:02d},{milissegundos:03d}"
    except Exception as e:
        logging.error(f"Erro ao formatar timestamp: {e}")
        return "00:00:00,000"

def processar_transcricao(caminho_video, nome_modelo, fila_progresso):
    """Processa a transcrição de áudio."""
    try:
        transcrever_audio(caminho_video, nome_modelo=nome_modelo)
        fila_progresso.put("Transcrição de áudio concluída!")
    except Exception as e:
        logging.error(f"Erro no processo de transcrição: {e}")
        fila_progresso.put(f"Erro na transcrição: {e}")

def main():
    parser = argparse.ArgumentParser(description="Processa vídeos para extrair frames com texto e gerar legendas com Whisper.")
    parser.add_argument("mascara_arquivos", type=str, help="Máscara de arquivos para processamento (ex: '*.mp4').")
    parser.add_argument("--modelo", type=str, default="medium", help="Modelo Whisper a ser utilizado (padrão: medium).")
    parser.add_argument("--recursivo", action="store_true", help="Busca recursivamente em subdiretórios.")
    parser.add_argument("--min_palavra", type=int, default=4, help="Tamanho mínimo das palavras a serem consideradas (padrão: 4).")
    args = parser.parse_args()

    mascara_arquivos = args.mascara_arquivos
    nome_modelo = args.modelo
    recursivo = args.recursivo
    min_palavra = args.min_palavra

    # Encontrar arquivos de vídeo para processar
    arquivos_video = glob.glob(mascara_arquivos, recursive=recursivo)

    if not arquivos_video:
        logging.error(f"Nenhum arquivo encontrado com a máscara: {mascara_arquivos}")
        return

    tempo_inicio = time()

    for caminho_video in arquivos_video:
        try:
            # Pasta de saída para frames com texto
            pasta_video = os.path.dirname(caminho_video)
            nome_video = os.path.splitext(os.path.basename(caminho_video))[0]
            pasta_saida = os.path.join(pasta_video, f"frames_{nome_video}")
            os.makedirs(pasta_saida, exist_ok=True)

            logging.info(f"Processando vídeo: {caminho_video}")
            logging.info(f"Pasta de saída: {pasta_saida}")

            # Filas para monitorar progresso
            fila_progresso_transcricao = Queue()

            # Criar e iniciar processos separados
            processo_frames = threading.Thread(target=processar_frames_com_opencv, args=(caminho_video, pasta_saida, 4, min_palavra))
            processo_transcricao = threading.Thread(target=processar_transcricao, args=(caminho_video, nome_modelo, fila_progresso_transcricao))

            processo_frames.start()
            processo_transcricao.start()

            # Monitorar progresso
            while processo_frames.is_alive() or processo_transcricao.is_alive():
                if not fila_progresso_transcricao.empty():
                    msg_transcricao = fila_progresso_transcricao.get()
                    logging.info(f"Transcrição: {msg_transcricao}")

            # Garantir que ambos os processos foram concluídos
            processo_frames.join()
            processo_transcricao.join()
        except Exception as e:
            logging.error(f"Erro ao processar vídeo {caminho_video}: {e}")

    tempo_total = time() - tempo_inicio
    logging.info(f"Processo concluído em {tempo_total:.2f} segundos.")

if __name__ == "__main__":
    main()
