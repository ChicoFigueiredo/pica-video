import os
import re
import shutil
import subprocess
import tempfile
import whisper
import argparse
from tqdm import tqdm
from time import time
from PIL import Image
import pytesseract
from multiprocessing import Process, Queue, Pool
import glob
from spellchecker import SpellChecker
from langdetect import detect, DetectorFactory
import re

# Fixar a seed para resultados consistentes na detecção de idioma
DetectorFactory.seed = 0

# Inicializa o corretor ortográfico
spell = SpellChecker(language="pt")

def extract_frames(video_path, temp_dir, fps=4, resolution="1280:720"):
    """Executa o ffmpeg para extrair frames com informações de tempo e coleta em memória."""
    
    frame_pattern = os.path.join(temp_dir, "frame_%06d.png")
    
    command = [
        "ffmpeg", "-i", video_path, "-vf", f"fps={fps},scale={resolution},showinfo", frame_pattern
    ]
    
    process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True)
    
    log_data = []
    
    try:
        for line in tqdm(process.stderr, desc="Extraindo frames", unit="line"):
            if "pts_time" in line:
                log_data.append(line)
    except Exception as e:
        raise RuntimeError(f"Erro ao extrair frames: {e}")

    process.wait()

    if process.returncode != 0:
        raise RuntimeError("Erro ao processar o vídeo com ffmpeg.")
    
    return log_data

def parse_log_data(log_data):
    """Processa os logs de ffmpeg para obter timestamps de cada frame."""
    
    frame_times = []
    for line in log_data:
        match = re.search(r'pts_time:([0-9.]+)', line)
        if match:
            timestamp = float(match.group(1))
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            milliseconds = int((timestamp - int(timestamp)) * 1000)
            frame_times.append((minutes, seconds, milliseconds))
    
    return frame_times

def rename_frames(frame_times, temp_dir):
    """Renomeia os frames de acordo com os tempos extraídos."""
    
    for i, (minutes, seconds, milliseconds) in enumerate(tqdm(frame_times, desc="Renomeando frames", unit="frame")):
        original_name = os.path.join(temp_dir, f"frame_{i+1:06d}.png")
        new_name = os.path.join(temp_dir, f"frame_{minutes:02d}-{seconds:02d}-{milliseconds:03d}.png")
        
        if os.path.exists(original_name):
            os.rename(original_name, new_name)
        else:
            print(f"Arquivo {original_name} não encontrado. Pulando...")


def is_human_readable_text(text):
    """Verifica se o texto é legível e está em português."""
    
    # Remover caracteres que não sejam letras ou espaços
    cleaned_text = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ\s]', '', text)
    
    # Remover espaços extras
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Verificar se ainda há texto após limpeza
    if not cleaned_text:
        return False
    
    # Tentar detectar o idioma
    try:
        language = detect(cleaned_text)
    except:
        # Se a detecção falhar, considerar como não legível
        return False
    
    # Verificar se o idioma é português
    if language != 'pt':
        return False
    
    # Dividir o texto em palavras
    words = cleaned_text.split()
    
    # Palavras com 4 letras ou mais
    long_words = [word for word in words if len(word) >= 4]
    
    # Critério: Pelo menos uma palavra com 4 letras ou mais
    if len(long_words) > 0:
        return True
    
    return False

def process_frame_with_ocr(args):
    """Processa um único frame com OCR e salva o texto."""
    frame_path, output_dir = args
    try:
        image = Image.open(frame_path)
        
        # Pré-processamento: Converter para escala de cinza
        image = image.convert("L")
        
        # Extrair texto com Tesseract, especificando o idioma português
        text = pytesseract.image_to_string(image, lang='por')
        
        if text.strip() and is_human_readable_text(text):
            # Mover o frame para a nova pasta
            output_frame_path = os.path.join(output_dir, os.path.basename(frame_path))
            shutil.copy(frame_path, output_frame_path)
            
            # Salvar o texto capturado em um arquivo txt com o mesmo nome do frame
            text_output_path = os.path.join(output_dir, os.path.basename(frame_path).replace(".png", ".txt"))
            with open(text_output_path, "w") as text_file:
                text_file.write(text.strip())
        return True  # Sucesso ao processar o frame
    except Exception as e:
        print(f"Erro ao processar o frame {frame_path}: {e}")
        return False  # Falha ao processar o frame

def detect_text_in_frames(temp_dir, output_dir, batch_size=5):
    """Faz varredura nos frames da pasta temporária em lotes e identifica aqueles que contêm texto."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".png")]

    # Prepara os argumentos para o Pool
    args_list = [(frame, output_dir) for frame in frame_files]

    # Processar os frames em paralelo usando multiprocessing Pool
    with Pool(processes=batch_size) as pool:
        results = list(tqdm(pool.imap_unordered(process_frame_with_ocr, args_list), 
                            total=len(frame_files), desc="Detectando texto nos frames", unit="frame"))

    print(f"Frames com texto foram movidos para: {output_dir}")
    return output_dir

def process_frames(video_path, temp_dir, progress_queue, batch_size):
    """Processo separado para lidar com extração e OCR dos frames."""
    try:
        log_data = extract_frames(video_path, temp_dir)
        frame_times = parse_log_data(log_data)
        rename_frames(frame_times, temp_dir)

        # Criar a pasta de saída para os frames com texto
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).replace(".mp4", "")
        output_dir = os.path.join(video_dir, f"frames_{video_name}")

        # Detectar os frames com texto e salvá-los na pasta de saída
        detect_text_in_frames(temp_dir, output_dir, batch_size=batch_size)

    except Exception as e:
        progress_queue.put(f"Erro ao processar frames: {e}")
    finally:
        progress_queue.put("Processamento de frames concluído!")

def transcribe_audio(video_path, model_name="medium", language="Portuguese"):
    """Processa o áudio do vídeo com o Whisper para gerar legendas e fala cronometrada."""
    
    whisper_model = whisper.load_model(model_name)
    
    result = whisper_model.transcribe(video_path, language=language, task="transcribe")
    
    srt_path = video_path.replace(".mp4", ".srt")
    cronometed_speech_path = video_path.replace(".mp4", "-Fala.Cronometrada.txt")
    
    try:
        with open(srt_path, "w") as srt_file:
            for segment in result['segments']:
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text'].strip()

                srt_file.write(f"{segment['id']}\n")
                srt_file.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                srt_file.write(f"{text}\n\n")
        
        with open(cronometed_speech_path, "w") as txt_file:
            for segment in result['segments']:
                start_time = format_timestamp(segment['start'])
                text = segment['text'].strip()
                txt_file.write(f"{start_time}: {text}\n")
    
    except Exception as e:
        print(f"Erro ao salvar arquivos de saída: {e}")
    
    print(f"Arquivos gerados: {srt_path}, {cronometed_speech_path}")

def format_timestamp(seconds):
    """Formata os segundos no formato de timestamp para SRT (hh:mm:ss,milliseconds)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def process_transcription(video_path, model_name, progress_queue):
    """Processo separado para lidar com a transcrição de áudio."""
    try:
        transcribe_audio(video_path, model_name=model_name)
    except Exception as e:
        progress_queue.put(f"Erro ao transcrever áudio: {e}")
    finally:
        progress_queue.put("Transcrição de áudio concluída!")

def find_files_with_mask(mask, recursive):
    """Encontra arquivos com base na máscara fornecida, com suporte a varredura recursiva."""
    if recursive:
        return glob.glob(mask, recursive=True)
    else:
        return glob.glob(mask)

def main():
    parser = argparse.ArgumentParser(description="Processa vídeos para extrair frames e gerar legendas com Whisper.")
    parser.add_argument("file_mask", type=str, help="Máscara de arquivos para processamento (ex: *.mp4).")
    parser.add_argument("--model", type=str, default="medium", help="Modelo Whisper a ser utilizado (padrão: medium).")
    parser.add_argument("--recursive", action="store_true", help="Se ativado, varre subpastas.")
    parser.add_argument("--batches", type=int, default=5, help="Número de lotes (padrão: 5).")
    parser.add_argument("--temp_dir", type=str, help="Pasta temporária para armazenar frames.")
    args = parser.parse_args()

    file_mask = args.file_mask
    model_name = args.model
    recursive = args.recursive
    batch_size = args.batches

    # Se temp_dir não for fornecido, utiliza o diretório temporário do sistema operacional
    if args.temp_dir:
        temp_dir_base = args.temp_dir
    else:
        temp_dir_base = tempfile.gettempdir()  # Diretório temporário padrão do sistema

    # Encontrar arquivos de acordo com a máscara fornecida
    video_files = find_files_with_mask(file_mask, recursive)

    if not video_files:
        print(f"Nenhum arquivo encontrado com a máscara: {file_mask}")
        return

    try:
        start_time = time()

        for video_path in video_files:
            temp_dir = os.path.join(temp_dir_base, os.path.basename(video_path).replace(".mp4", ""))
            os.makedirs(temp_dir, exist_ok=True)

            print(f"Usando diretório temporário: {temp_dir} para o vídeo: {video_path}")

            # Criar filas para monitorar o progresso de ambos os processos
            progress_queue_frames = Queue()
            progress_queue_transcription = Queue()

            # Criar e iniciar processos separados
            frame_process = Process(target=process_frames, args=(video_path, temp_dir, progress_queue_frames, batch_size))
            transcription_process = Process(target=process_transcription, args=(video_path, model_name, progress_queue_transcription))

            frame_process.start()
            transcription_process.start()

            # Monitorar o progresso dos dois processos
            while frame_process.is_alive() or transcription_process.is_alive():
                if not progress_queue_frames.empty():
                    msg_frames = progress_queue_frames.get()
                    print(f"Frames: {msg_frames}")
                
                if not progress_queue_transcription.empty():
                    msg_transcription = progress_queue_transcription.get()
                    print(f"Transcrição: {msg_transcription}")

            # Certificar-se de que ambos os processos foram concluídos
            frame_process.join()
            transcription_process.join()

        total_time = time() - start_time
        print(f"\nProcesso concluído em {total_time:.2f} segundos.")

    except Exception as e:
        print(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    main()
