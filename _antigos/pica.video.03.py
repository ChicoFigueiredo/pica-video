import os
import re
import shutil
import subprocess
import tempfile
import whisper
import argparse
import glob
import logging
from tqdm import tqdm
from time import time
from PIL import Image, ImageFilter
import pytesseract
from multiprocessing import Process, Queue, Pool
from spellchecker import SpellChecker
from langdetect import detect, DetectorFactory

# Fixar a seed para resultados consistentes na detecção de idioma
DetectorFactory.seed = 0

# Inicializar o corretor ortográfico para português
corretor = SpellChecker(language="pt")

# Configurar o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def extrair_frames(caminho_video, pasta_temp, fps=4, resolucao="1280:720"):
    """Extrai frames do vídeo usando ffmpeg e coleta timestamps."""
    try:
        padrao_frame = os.path.join(pasta_temp, "frame_%06d.png")
        comando = [
            "ffmpeg", "-i", caminho_video, "-vf", f"fps={fps},scale={resolucao},showinfo", padrao_frame
        ]

        processo = subprocess.Popen(comando, stderr=subprocess.PIPE, text=True)
        dados_log = []

        for linha in tqdm(processo.stderr, desc="Extraindo frames", unit="linha"):
            if "pts_time" in linha:
                dados_log.append(linha)

        processo.wait()

        if processo.returncode != 0:
            raise RuntimeError("Erro ao processar o vídeo com ffmpeg.")

        return dados_log
    except Exception as e:
        logging.error(f"Erro ao extrair frames: {e}")
        raise

def analisar_dados_log(dados_log):
    """Analisa os logs do ffmpeg para obter timestamps de cada frame."""
    tempos_frames = []
    try:
        for linha in dados_log:
            match = re.search(r'pts_time:([0-9.]+)', linha)
            if match:
                timestamp = float(match.group(1))
                minutos = int(timestamp // 60)
                segundos = int(timestamp % 60)
                milissegundos = int((timestamp - int(timestamp)) * 1000)
                tempos_frames.append((minutos, segundos, milissegundos))
    except Exception as e:
        logging.error(f"Erro ao analisar dados do log: {e}")
        raise

    return tempos_frames

def renomear_frames(tempos_frames, pasta_temp):
    """Renomeia frames com base nos timestamps extraídos."""
    try:
        for i, (minutos, segundos, milissegundos) in enumerate(tqdm(tempos_frames, desc="Renomeando frames", unit="frame")):
            nome_original = os.path.join(pasta_temp, f"frame_{i+1:06d}.png")
            nome_novo = os.path.join(pasta_temp, f"frame_{minutos:02d}-{segundos:02d}-{milissegundos:03d}.png")

            if os.path.exists(nome_original):
                os.rename(nome_original, nome_novo)
            else:
                logging.warning(f"Arquivo {nome_original} não encontrado. Pulando...")
    except Exception as e:
        logging.error(f"Erro ao renomear frames: {e}")
        raise

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
        imagem = imagem.resize((imagem.width * 2, imagem.height * 2), Image.ANTIALIAS)

        return imagem
    except Exception as e:
        logging.error(f"Erro ao preprocessar imagem: {e}")
        raise

def texto_legivel(texto):
    """Verifica se o texto é legível e está em português."""
    try:
        # Remover caracteres não alfabéticos
        texto_limpo = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ\s]', '', texto)
        texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()

        if not texto_limpo:
            return False

        # Detectar idioma
        idioma = detect(texto_limpo)
        if idioma != 'pt':
            return False

        # Dividir em palavras
        palavras = texto_limpo.split()

        # Palavras com 4 letras ou mais
        palavras_longas = [palavra for palavra in palavras if len(palavra) >= 4]

        if len(palavras_longas) > 0:
            return True

        return False
    except Exception as e:
        logging.error(f"Erro na validação do texto: {e}")
        return False

def processar_frame_ocr(args):
    """Processa um frame individual com OCR e salva o texto."""
    caminho_frame, pasta_saida = args
    try:
        imagem = Image.open(caminho_frame)

        # Preprocessar a imagem
        imagem = preprocessar_imagem(imagem)

        # Configurar Tesseract para maior precisão
        configuracao_tesseract = r'--oem 1 --psm 6 -l por'

        # Executar OCR
        texto = pytesseract.image_to_string(imagem, config=configuracao_tesseract)

        if texto.strip() and texto_legivel(texto):
            # Copiar o frame para a pasta de saída
            caminho_frame_saida = os.path.join(pasta_saida, os.path.basename(caminho_frame))
            shutil.copy(caminho_frame, caminho_frame_saida)

            # Salvar o texto extraído
            caminho_texto_saida = os.path.join(pasta_saida, os.path.basename(caminho_frame).replace(".png", ".txt"))
            with open(caminho_texto_saida, "w", encoding="utf-8") as arquivo_texto:
                arquivo_texto.write(texto.strip())
    except Exception as e:
        logging.error(f"Erro ao processar frame {caminho_frame}: {e}")

def detectar_texto_frames(pasta_temp, pasta_saida, tamanho_lote=5):
    """Verifica frames e identifica aqueles que contêm texto."""
    try:
        if not os.path.exists(pasta_saida):
            os.makedirs(pasta_saida)

        arquivos_frames = [os.path.join(pasta_temp, f) for f in os.listdir(pasta_temp) if f.endswith(".png")]

        args_list = [(frame, pasta_saida) for frame in arquivos_frames]

        with Pool(processes=tamanho_lote) as pool:
            list(tqdm(pool.imap_unordered(processar_frame_ocr, args_list),
                      total=len(arquivos_frames), desc="Processando frames", unit="frame"))
    except Exception as e:
        logging.error(f"Erro ao detectar texto nos frames: {e}")
        raise

def processar_frames(caminho_video, pasta_temp, fila_progresso, tamanho_lote):
    """Processa frames: extração, renomeação e OCR."""
    try:
        dados_log = extrair_frames(caminho_video, pasta_temp)
        tempos_frames = analisar_dados_log(dados_log)
        renomear_frames(tempos_frames, pasta_temp)

        # Pasta de saída para frames com texto
        pasta_video = os.path.dirname(caminho_video)
        nome_video = os.path.splitext(os.path.basename(caminho_video))[0]
        pasta_saida = os.path.join(pasta_video, f"frames_{nome_video}")

        detectar_texto_frames(pasta_temp, pasta_saida, tamanho_lote=tamanho_lote)
        fila_progresso.put("Processamento de frames concluído!")
    except Exception as e:
        logging.error(f"Erro ao processar frames para {caminho_video}: {e}")
        fila_progresso.put(f"Erro ao processar frames: {e}")
    finally:
        # Limpar pasta temporária
        try:
            shutil.rmtree(pasta_temp)
            logging.info(f"Pasta temporária {pasta_temp} removida.")
        except Exception as e:
            logging.warning(f"Não foi possível remover a pasta temporária {pasta_temp}: {e}")

def transcrever_audio(caminho_video, nome_modelo="medium", idioma="Portuguese"):
    """Transcreve áudio do vídeo usando o Whisper."""
    try:
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

def encontrar_arquivos_mascara(mascara, recursivo):
    """Encontra arquivos com base na máscara fornecida."""
    try:
        if recursivo:
            return glob.glob(mascara, recursive=True)
        else:
            return glob.glob(mascara)
    except Exception as e:
        logging.error(f"Erro ao encontrar arquivos com a máscara {mascara}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Processa vídeos para extrair frames e gerar legendas com Whisper.")
    parser.add_argument("mascara_arquivos", type=str, help="Máscara de arquivos para processamento (ex: *.mp4).")
    parser.add_argument("--modelo", type=str, default="medium", help="Modelo Whisper a ser utilizado (padrão: medium).")
    parser.add_argument("--recursivo", action="store_true", help="Busca recursivamente em subdiretórios.")
    parser.add_argument("--lotes", type=int, default=5, help="Número de lotes paralelos (padrão: 5).")
    parser.add_argument("--pasta_temp", type=str, help="Pasta temporária para armazenar frames.")
    args = parser.parse_args()

    mascara_arquivos = args.mascara_arquivos
    nome_modelo = args.modelo
    recursivo = args.recursivo
    tamanho_lote = args.lotes

    # Usar pasta temporária do sistema se pasta_temp não for fornecida
    if args.pasta_temp:
        pasta_temp_base = args.pasta_temp
    else:
        pasta_temp_base = tempfile.gettempdir()

    # Encontrar arquivos de vídeo para processar
    arquivos_video = encontrar_arquivos_mascara(mascara_arquivos, recursivo)

    if not arquivos_video:
        logging.error(f"Nenhum arquivo encontrado com a máscara: {mascara_arquivos}")
        return

    tempo_inicio = time()

    for caminho_video in arquivos_video:
        try:
            pasta_temp = os.path.join(pasta_temp_base, os.path.splitext(os.path.basename(caminho_video))[0])
            os.makedirs(pasta_temp, exist_ok=True)

            logging.info(f"Processando vídeo: {caminho_video}")
            logging.info(f"Pasta temporária: {pasta_temp}")

            # Filas para monitorar progresso
            fila_progresso_frames = Queue()
            fila_progresso_transcricao = Queue()

            # Criar e iniciar processos separados
            processo_frames = Process(target=processar_frames, args=(caminho_video, pasta_temp, fila_progresso_frames, tamanho_lote))
            processo_transcricao = Process(target=processar_transcricao, args=(caminho_video, nome_modelo, fila_progresso_transcricao))

            processo_frames.start()
            processo_transcricao.start()

            # Monitorar progresso
            while processo_frames.is_alive() or processo_transcricao.is_alive():
                if not fila_progresso_frames.empty():
                    msg_frames = fila_progresso_frames.get()
                    logging.info(f"Frames: {msg_frames}")

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
