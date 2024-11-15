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

def extract_frames(video_path, temp_dir, fps=4, resolution="1280:720"):
    """Executa o ffmpeg para extrair frames com informações de tempo e coleta em memória."""
    
    # Definir caminho temporário para os frames
    frame_pattern = os.path.join(temp_dir, "frame_%06d.png")
    
    # Comando ffmpeg com filtro showinfo
    command = [
        "ffmpeg", "-i", video_path, "-vf", f"fps={fps},scale={resolution},showinfo", frame_pattern
    ]
    
    # Rodar o ffmpeg e capturar stdout/stderr
    process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True)
    
    # Variáveis para armazenar frames e logs
    log_data = []
    
    # Leitura do stderr para capturar os logs de showinfo
    try:
        for line in tqdm(process.stderr, desc="Extraindo frames", unit="line"):
            if "pts_time" in line:
                log_data.append(line)  # Armazena linhas com informações de tempo
    except Exception as e:
        raise RuntimeError(f"Erro ao extrair frames: {e}")

    process.wait()  # Aguarda o fim do processo

    # Verifica se o processo foi bem-sucedido
    if process.returncode != 0:
        raise RuntimeError("Erro ao processar o vídeo com ffmpeg.")
    
    return log_data

def parse_log_data(log_data):
    """Processa os logs de ffmpeg para obter timestamps de cada frame."""
    
    frame_times = []
    for line in log_data:
        # Expressão regular para capturar o tempo pts_time
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

def transcribe_audio(video_path, model_name="medium", language="Portuguese"):
    """Processa o áudio do vídeo com o Whisper para gerar legendas e fala cronometrada."""
    
    try:
        # Carregar o modelo Whisper
        whisper_model = whisper.load_model(model_name)
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o modelo Whisper: {e}")
    
    try:
        # Transcrever o vídeo
        result = whisper_model.transcribe(video_path, language=language, task="transcribe")
    except Exception as e:
        raise RuntimeError(f"Erro ao transcrever o vídeo: {e}")
    
    # Gera o arquivo SRT para legendas
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
        
        # Gera o arquivo de fala cronometrada
        with open(cronometed_speech_path, "w") as txt_file:
            for segment in result['segments']:
                start_time = format_timestamp(segment['start'])
                text = segment['text'].strip()
                txt_file.write(f"{start_time}: {text}\n")
    
    except Exception as e:
        raise RuntimeError(f"Erro ao salvar arquivos de saída: {e}")
    
    print(f"Arquivos gerados: {srt_path}, {cronometed_speech_path}")

def format_timestamp(seconds):
    """Formata os segundos no formato de timestamp para SRT (hh:mm:ss,milliseconds)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def detect_text_in_frames(temp_dir):
    """Faz varredura nos frames da pasta temporária e identifica aqueles que contêm texto."""
    
    # Cria uma nova pasta temporária para armazenar os frames com texto
    text_frames_dir = tempfile.mkdtemp(prefix="frames_com_texto_")
    
    # Varre todos os frames na pasta temporária
    for frame_file in tqdm(os.listdir(temp_dir), desc="Detectando texto nos frames", unit="frame"):
        frame_path = os.path.join(temp_dir, frame_file)
        if frame_file.endswith(".png"):
            try:
                # Carrega a imagem do frame
                image = Image.open(frame_path)
                
                # Usa o Tesseract OCR para detectar texto
                text = pytesseract.image_to_string(image)
                
                # Se encontrar texto, move o frame para a nova pasta
                if text.strip():  # Se o texto não estiver vazio
                    shutil.copy(frame_path, text_frames_dir)
            except Exception as e:
                print(f"Erro ao processar o frame {frame_file}: {e}")
    
    print(f"Frames com texto foram movidos para: {text_frames_dir}")
    return text_frames_dir

def main():
    # Configuração do argparse para receber argumentos do prompt de comando
    parser = argparse.ArgumentParser(description="Processa vídeo para extrair frames e gerar legendas com Whisper.")
    parser.add_argument("video_path", type=str, help="Caminho do arquivo de vídeo.")
    parser.add_argument("--model", type=str, default="medium", help="Modelo Whisper a ser utilizado (padrão: medium).")
    args = parser.parse_args()

    video_path = args.video_path
    model_name = args.model
    
    # Checar se o arquivo de vídeo existe
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Arquivo de vídeo '{video_path}' não encontrado.")
    
    try:
        start_time = time()

        # Criar um diretório temporário exclusivo para este vídeo
        with tempfile.TemporaryDirectory(dir="/mnt/t/temp", prefix=os.path.basename(video_path).replace(".mp4", "") + "_") as temp_dir:
            print(f"Usando diretório temporário: {temp_dir}")

            # Passo 1: Extrair frames e coletar dados de tempo em memória
            log_data = extract_frames(video_path, temp_dir)

            # Passo 2: Analisar o log para extrair os tempos de cada frame
            frame_times = parse_log_data(log_data)

            # Passo 3: Renomear os frames de acordo com o timestamp extraído
            rename_frames(frame_times, temp_dir)

            # Passo 4: Processar a fala com Whisper e gerar arquivos SRT e texto cronometrado
            transcribe_audio(video_path, model_name=model_name)

            # Passo 5: Detectar texto nos frames e movê-los para uma nova pasta
            detect_text_in_frames(temp_dir)

        # O diretório temporário será automaticamente removido aqui

        # Exibir o tempo total de execução
        total_time = time() - start_time
        print(f"\nProcesso concluído em {total_time:.2f} segundos.")

    except Exception as e:
        print(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    main()
