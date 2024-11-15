import os
import re
import cv2  # Importar o OpenCV
import whisper
import argparse
import logging
import threading
import glob
import warnings
from tqdm import tqdm
from time import time
from PIL import Image, ImageFilter
import pytesseract
from langdetect import detect, DetectorFactory
from spellchecker import SpellChecker
from queue import Queue

# Suprimir avisos específicos
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# Fixar a seed para resultados consistentes na detecção de idioma
DetectorFactory.seed = 0

# Inicializar o corretor ortográfico para português
corretor = SpellChecker(language="pt")

# Configurar o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Compatibilidade com diferentes versões do Pillow
try:
    resampling_filter = Image.Resampling.LANCZOS
except AttributeError:
    resampling_filter = Image.LANCZOS

def preprocessar_imagem(imagem):
    """Preprocessa a imagem para melhorar a precisão do OCR."""
    try:
        # Converter para escala de cinza
        imagem = imagem.convert("L")

        # Aplicar limiarização
        imagem = imagem.point(lambda x: 0 if x < 140 else 255, '1')

        # Aplicar filtro de mediana para reduzir ruído
        imagem = imagem.filter(ImageFilter.MedianFilter())

        # Redimensionar a imagem para o dobro do tamanho
        imagem = imagem.resize((imagem.width * 2, imagem.height * 2), resampling_filter)

        return imagem
    except Exception as e:
        logging.error(f"Erro ao preprocessar imagem: {e}")
        raise

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

def processar_frames_com_opencv(caminho_video, pasta_saida, fps_extracao=4, min_palavra=4):
    """Processa os frames do vídeo usando OpenCV e salva apenas aqueles com pelo menos uma palavra legível."""
    try:
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
                    # Converter o frame para RGB (OpenCV lê em BGR)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Converter para PIL Image
                    imagem = Image.fromarray(frame_rgb)

                    # Preprocessar a imagem
                    imagem_proc = preprocessar_imagem(imagem)

                    # Configurar Tesseract para maior precisão
                    configuracao_tesseract = (
                        r'--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -l por'
                    )

                    # Executar OCR
                    texto = pytesseract.image_to_string(imagem_proc, config=configuracao_tesseract)

                    if texto.strip() and texto_legivel(texto, min_palavra):
                        # Formatar o timestamp em HH_MM_SS.FFFF
                        timestamp_formatado = formatar_timestamp_para_nome(timestamp_ms)

                        # Salvar o frame com texto
                        nome_frame = f"frame_{timestamp_formatado}.png"
                        caminho_frame_saida = os.path.join(pasta_saida, nome_frame)
                        imagem.save(caminho_frame_saida)

                        # Salvar o texto extraído
                        caminho_texto_saida = os.path.join(pasta_saida, f"frame_{timestamp_formatado}.txt")
                        with open(caminho_texto_saida, "w", encoding="utf-8") as arquivo_texto:
                            arquivo_texto.write(texto.strip())
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
    parser = argparse.ArgumentParser(description="Processa vídeos para extrair frames e gerar legendas com Whisper.")
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
