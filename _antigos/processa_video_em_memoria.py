import os
import re
import whisper
import argparse
import logging
import subprocess
import threading
from tqdm import tqdm
from time import time
from PIL import Image, ImageFilter
import pytesseract
from io import BytesIO
from langdetect import detect, DetectorFactory
from spellchecker import SpellChecker

# Fixar a seed para resultados consistentes na detecção de idioma
DetectorFactory.seed = 0

# Inicializar o corretor ortográfico para português
corretor = SpellChecker(language="pt")

# Configurar o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

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

def processar_frames_em_memoria(caminho_video, pasta_saida):
    """Processa os frames do vídeo em memória e salva apenas aqueles com texto legível."""
    try:
        # Comando do ffmpeg para extrair frames via pipe
        comando = [
            'ffmpeg', '-i', caminho_video,
            '-f', 'image2pipe',
            '-vcodec', 'png',
            '-vf', 'fps=4,scale=1280:720',
            '-loglevel', 'error',  # Reduz a verbosidade
            'pipe:1'
        ]

        # Iniciar o processo do ffmpeg
        processo = subprocess.Popen(comando, stdout=subprocess.PIPE, bufsize=10**8)

        frame_num = 0
        pbar = tqdm(desc="Processando frames", unit="frame")

        while True:
            # Ler o tamanho do próximo frame
            data = processo.stdout.read(8)
            if not data:
                break

            # Converter os bytes lidos em tamanho do frame
            # PNGs não têm tamanho fixo, então usamos BytesIO
            # Recolocar o cabeçalho lido de volta nos dados
            data += processo.stdout.read(1024 * 1024)  # Ler até 1MB por vez

            imagem = Image.open(BytesIO(data))

            # Preprocessar a imagem
            imagem_proc = preprocessar_imagem(imagem)

            # Configurar Tesseract para maior precisão
            configuracao_tesseract = r'--oem 1 --psm 6 -l por'

            # Executar OCR
            texto = pytesseract.image_to_string(imagem_proc, config=configuracao_tesseract)

            if texto.strip() and texto_legivel(texto):
                # Salvar o frame com texto
                nome_frame = f"frame_{frame_num:06d}.png"
                caminho_frame_saida = os.path.join(pasta_saida, nome_frame)
                imagem.save(caminho_frame_saida)

                # Salvar o texto extraído
                caminho_texto_saida = os.path.join(pasta_saida, f"frame_{frame_num:06d}.txt")
                with open(caminho_texto_saida, "w", encoding="utf-8") as arquivo_texto:
                    arquivo_texto.write(texto.strip())

            frame_num += 1
            pbar.update(1)

        pbar.close()
        processo.terminate()
    except Exception as e:
        logging.error(f"Erro ao processar frames em memória: {e}")
        raise

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

def main():
    parser = argparse.ArgumentParser(description="Processa vídeos para extrair frames e gerar legendas com Whisper.")
    parser.add_argument("mascara_arquivos", type=str, help="Máscara de arquivos para processamento (ex: *.mp4).")
    parser.add_argument("--modelo", type=str, default="medium", help="Modelo Whisper a ser utilizado (padrão: medium).")
    parser.add_argument("--recursivo", action="store_true", help="Busca recursivamente em subdiretórios.")
    args = parser.parse_args()

    mascara_arquivos = args.mascara_arquivos
    nome_modelo = args.modelo
    recursivo = args.recursivo

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
            processo_frames = threading.Thread(target=processar_frames_em_memoria, args=(caminho_video, pasta_saida))
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
