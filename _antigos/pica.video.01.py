import os
import re
import shutil
import subprocess
import tempfile
import warnings
import argparse
import glob
import logging
from tqdm import tqdm
from time import time
from multiprocessing import Process, Queue
from faster_whisper import WhisperModel

# Configurar o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def extrair_frames(caminho_video, pasta_saida, fps=4, resolucao="1280:720"):
    """Extrai frames do vídeo usando ffmpeg e coleta timestamps."""
    try:
        padrao_frame = os.path.join(pasta_saida, "frame_%06d.png")
        comando = [
            "ffmpeg", "-i", caminho_video, "-vf", f"fps={fps},scale={resolucao},showinfo", padrao_frame
        ]

        processo = subprocess.Popen(comando, stderr=subprocess.PIPE, text=True)
        dados_log = []

        for linha in tqdm(processo.stderr, desc="Extraindo frames", unit="linha"):
            if "pts_time" in linha:
                dados_log.append(linha)

        processo.wait(timeout=300)  # Timeout de 5 minutos

        if processo.returncode != 0:
            raise RuntimeError("Erro ao processar o vídeo com ffmpeg. Código de retorno diferente de zero.")

        return dados_log
    except subprocess.TimeoutExpired:
        logging.error(f"Processo do ffmpeg excedeu o tempo limite para {caminho_video}.")
        raise
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

def renomear_frames(tempos_frames, pasta_saida, nome_base):
    """Renomeia frames com base nos timestamps extraídos."""
    try:
        for i, (minutos, segundos, milissegundos) in enumerate(tqdm(tempos_frames, desc="Renomeando frames", unit="frame")):
            nome_original = os.path.join(pasta_saida, f"frame_{i+1:06d}.png")
            nome_novo = os.path.join(pasta_saida, f"frame_{nome_base}_{minutos:02d}-{segundos:02d}-{milissegundos:03d}.png")

            if os.path.exists(nome_original):
                os.rename(nome_original, nome_novo)
            else:
                logging.warning(f"Arquivo {nome_original} não encontrado. Pulando...")
    except Exception as e:
        logging.error(f"Erro ao renomear frames: {e}")
        raise

def processar_frames(caminho_video, pasta_saida, fila_progresso):
    """Processa frames: extração e renomeação."""
    try:
        # Extrair nome base do vídeo
        nome_base = os.path.splitext(os.path.basename(caminho_video))[0]

        # Extrair frames
        dados_log = extrair_frames(caminho_video, pasta_saida)
        tempos_frames = analisar_dados_log(dados_log)

        # Renomear frames
        renomear_frames(tempos_frames, pasta_saida, nome_base)

        fila_progresso.put("Processamento de frames concluído!")
    except Exception as e:
        logging.error(f"Erro ao processar frames para {caminho_video}: {e}")
        fila_progresso.put(f"Erro ao processar frames: {e}")

def transcrever_audio_faster_whisper(caminho_video, nome_modelo="large-v3", idioma="pt"):
    """Transcreve áudio do vídeo usando o Faster-Whisper."""
    try:
        logging.info(f"Iniciando Transcrição do vídeo usando o Faster-Whisper")

        # Especificar o caminho ou tamanho do modelo para o WhisperModel
        modelo_whisper = WhisperModel(model_size_or_path=nome_modelo)  # Carregar o modelo com faster-whisper
        segmentos, _ = modelo_whisper.transcribe(caminho_video, language=idioma)

        caminho_srt = caminho_video.replace(".mp4", ".srt")
        caminho_fala_cronometrada = caminho_video.replace(".mp4", "-Fala.Cronometrada.txt")

        logging.info(f"Salvando SRT")
        with open(caminho_srt, "w", encoding="utf-8") as arquivo_srt:
            for segmento in segmentos:
                inicio = segmento.start
                fim = segmento.end
                texto = segmento.text.strip()

                arquivo_srt.write(f"{segmento.id}\n")
                arquivo_srt.write(f"{formatar_timestamp(inicio)} --> {formatar_timestamp(fim)}\n")
                arquivo_srt.write(f"{texto}\n\n")

        logging.info(f"Salvando Fala Cronometrada")
        with open(caminho_fala_cronometrada, "w", encoding="utf-8") as arquivo_txt:
            for segmento in segmentos:
                inicio = formatar_timestamp(segmento.start)
                texto = segmento.text.strip()
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
        transcrever_audio_faster_whisper(caminho_video, nome_modelo=nome_modelo)
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

def formatar_tempo_humano(tempo_segundos):
    """Formata o tempo em horas, minutos e segundos."""
    horas = int(tempo_segundos // 3600)
    minutos = int((tempo_segundos % 3600) // 60)
    segundos = int(tempo_segundos % 60)
    
    partes = []
    if horas > 0:
        partes.append(f"{horas} horas")
    if minutos > 0:
        partes.append(f"{minutos} minutos")
    if segundos > 0 or not partes:
        partes.append(f"{segundos} segundos")
    
    return ", ".join(partes)

def main():
    parser = argparse.ArgumentParser(description="Processa vídeos para extrair frames e gerar legendas com Faster-Whisper.")
    parser.add_argument("mascara_arquivos", type=str, help="Máscara de arquivos para processamento (ex: *.mp4).")
    parser.add_argument("--modelo", type=str, default="large-v3", help="Modelo Whisper a ser utilizado (padrão: small).")
    parser.add_argument("--recursivo", action="store_true", help="Busca recursivamente em subdiretórios.")
    parser.add_argument("--pasta_saida", type=str, help="Pasta de saída para armazenar frames.")
    args = parser.parse_args()

    mascara_arquivos = args.mascara_arquivos
    nome_modelo = args.modelo
    recursivo = args.recursivo

    # Usar a pasta do arquivo processado se pasta_saida não for fornecida
    for caminho_video in encontrar_arquivos_mascara(mascara_arquivos, recursivo):
        try:
            if args.pasta_saida:
                pasta_saida = args.pasta_saida
            else:
                pasta_saida = os.path.dirname(caminho_video)  # mesma pasta do arquivo processado

            pasta_saida = os.path.join(pasta_saida, os.path.splitext(os.path.basename(caminho_video))[0])
            os.makedirs(pasta_saida, exist_ok=True)

            logging.info(f"Processando vídeo: {caminho_video}")
            logging.info(f"Pasta de saída: {pasta_saida}")

            # Filas para monitorar progresso
            fila_progresso_frames = Queue()
            fila_progresso_transcricao = Queue()

            # Criar e iniciar processos separados
            processo_frames = Process(target=processar_frames, args=(caminho_video, pasta_saida, fila_progresso_frames))
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
    tempo_formatado = formatar_tempo_humano(tempo_total)
    logging.info(f"Processo concluído em {tempo_formatado}.")

if __name__ == "__main__":
    tempo_inicio = time()  # Capturar o tempo de início antes da execução principal
    main()
