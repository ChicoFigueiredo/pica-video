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
from multiprocessing import Process, Queue

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

def detect_text_in_frames(temp_dir, output_dir):
    """Faz varredura nos frames da pasta temporária e identifica aqueles que contêm texto."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for frame_file in tqdm(os.listdir(temp_dir), desc="Detectando texto nos frames", unit="frame"):
        frame_path = os.path.join(temp_dir, frame_file)
        if frame_file.endswith(".png"):
            try:
                image = Image.open(frame_path)
                text = pytesseract.image_to_string(image)
                
                if text.strip():
                    # Mover o frame para a nova pasta
                    output_frame_path = os.path.join(output_dir, frame_file)
                    shutil.copy(frame_path, output_frame_path)

                    # Salvar o texto capturado em um arquivo txt com o mesmo nome do frame
                    text_output_path = os.path.join(output_dir, frame_file.replace(".png", ".txt"))
                    with open(text_output_path, "w") as text_file:
                        text_file.write(text.strip())
            except Exception as e:
                print(f"Erro ao processar o frame {frame_file}: {e}")
    
    print(f"Frames com texto foram movidos para: {output_dir}")
    return output_dir

def process_frames(video_path, temp_dir, progress_queue):
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
        detect_text_in_frames(temp_dir, output_dir)

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

def main():
    parser = argparse.ArgumentParser(description="Processa vídeo para extrair frames e gerar legendas com Whisper.")
    parser.add_argument("video_path", type=str, help="Caminho do arquivo de vídeo.")
    parser.add_argument("--model", type=str, default="medium", help="Modelo Whisper a ser utilizado (padrão: medium).")
    args = parser.parse_args()

    video_path = args.video_path
    model_name = args.model

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Arquivo de vídeo '{video_path}' não encontrado.")
    
    try:
        start_time = time()

        with tempfile.TemporaryDirectory(dir="/mnt/t/temp", prefix=os.path.basename(video_path).replace(".mp4", "") + "_") as temp_dir:
            print(f"Usando diretório temporário: {temp_dir}")

            # Criar filas para monitorar o progresso de ambos os processos
            progress_queue_frames = Queue()
            progress_queue_transcription = Queue()

            # Criar e iniciar processos separados
            frame_process = Process(target=process_frames, args=(video_path, temp_dir, progress_queue_frames))
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
