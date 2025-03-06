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
from multiprocessing import Process, Queue, set_start_method
from faster_whisper import WhisperModel
import torch  # Import torch to check for GPU availability
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer  # Import for translation

# Import Tkinter modules
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Import ttkthemes for theming
from ttkthemes import ThemedTk

# Set multiprocessing start method to 'spawn'
set_start_method('spawn', force=True)

# Configurar o logging para verbosidade máxima
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

# Check CUDA version and device properties
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    device_name = torch.cuda.get_device_name(0)
    logging.info(f"CUDA Version: {cuda_version}")
    logging.info(f"CUDA Device: {device_name}")
else:
    logging.warning("CUDA is not available. The script will run on CPU.")

print("")
print("===============================================================================================")
print("")


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
        logging.error(f"Erro ao extrair frames: {e}", exc_info=True)
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

        # Send progress updates to the queue
        total_frames = len(tempos_frames)
        for i, (minutos, segundos, milissegundos) in enumerate(tqdm(tempos_frames, desc="Renomeando frames", unit="frame")):
            # ...existing renaming code...
            progress = (i + 1) / total_frames
            fila_progresso.put(progress)
        fila_progresso.put("Processamento de frames concluído!")
    except Exception as e:
        logging.error(f"Erro ao processar frames para {caminho_video}: {e}")
        fila_progresso.put(f"Erro ao processar frames: {e}")


def transcrever_audio_faster_whisper(caminho_audio, nome_modelo="large-v3", idioma=None, fila_progresso=None):
    """Transcreve áudio do vídeo ou arquivo MP3 usando o Faster-Whisper."""
    try:
        logging.info("Iniciando Transcrição do áudio usando o Faster-Whisper")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        modelo_whisper = WhisperModel(model_size_or_path=nome_modelo, device=device, compute_type="int8")
        logging.debug(f"Modelo {nome_modelo} carregado com sucesso.")

        base_path = caminho_audio.rsplit(".", 1)[0]
        detected_language = None
        arquivos = {}

        # Iniciar transcrição e obter gerador de segmentos
        resultado = modelo_whisper.transcribe(caminho_audio, beam_size=5, language=idioma)
        generator = resultado[0]
        info = resultado[1]

        detected_language = info.language
        logging.info(f"Linguagem detectada: {detected_language}")

        if detected_language == "en":
            # Arquivos para transcrição em inglês e português
            arquivos['srt_en'] = open(f"{base_path}-en.srt", "w", encoding="utf-8")
            arquivos['fala_cron_en'] = open(f"{base_path}-en-Fala.Cronometrada.txt", "w", encoding="utf-8")
            arquivos['srt'] = open(f"{base_path}.srt", "w", encoding="utf-8")
            arquivos['fala_cron'] = open(f"{base_path}-Fala.Cronometrada.txt", "w", encoding="utf-8")
            # Carregar modelo de tradução
            arquivos['tokenizer'] = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
            arquivos['translation_model'] = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B").to(device)
            arquivos['tokenizer'].src_lang = "en"
        else:
            # Arquivos para transcri��ão no idioma detectado
            arquivos['srt'] = open(f"{base_path}.srt", "w", encoding="utf-8")
            arquivos['fala_cron'] = open(f"{base_path}-Fala.Cronometrada.txt", "w", encoding="utf-8")

        segment_id = 1
        total_segments = len(list(modelo_whisper.transcribe(caminho_audio, beam_size=5, language=idioma)[0]))
        generator = modelo_whisper.transcribe(caminho_audio, beam_size=5, language=idioma)[0]
        segment_id = 1
        for segmento in generator:
            inicio = segmento.start
            fim = segmento.end
            texto = segmento.text.strip()

            if detected_language == "en":
                # Salvar segmento em inglês
                arquivos['srt_en'].write(f"{segment_id}\n")
                arquivos['srt_en'].write(f"{formatar_timestamp(inicio)} --> {formatar_timestamp(fim)}\n")
                arquivos['srt_en'].write(f"{texto}\n\n")
                arquivos['fala_cron_en'].write(f"{formatar_timestamp(inicio)}: {texto}\n")

                # Traduzir e salvar em português
                encoded = arquivos['tokenizer'](texto, return_tensors="pt").to(device)
                generated_tokens = arquivos['translation_model'].generate(**encoded, forced_bos_token_id=arquivos['tokenizer'].get_lang_id("pt"))
                texto_traduzido = arquivos['tokenizer'].batch_decode(generated_tokens, skip_special_tokens=True)[0]

                arquivos['srt'].write(f"{segment_id}\n")
                arquivos['srt'].write(f"{formatar_timestamp(inicio)} --> {formatar_timestamp(fim)}\n")
                arquivos['srt'].write(f"{texto_traduzido}\n\n")
                arquivos['fala_cron'].write(f"{formatar_timestamp(inicio)}: {texto_traduzido}\n")

                logging.info(f"Segmento {segment_id}: {formatar_timestamp(inicio)} --> {formatar_timestamp(fim)} {texto} | pt='{texto_traduzido}'")
            else:
                # Salvar segmento no idioma detectado
                arquivos['srt'].write(f"{segment_id}\n")
                arquivos['srt'].write(f"{formatar_timestamp(inicio)} --> {formatar_timestamp(fim)}\n")
                arquivos['srt'].write(f"{texto}\n\n")
                arquivos['fala_cron'].write(f"{formatar_timestamp(inicio)}: {texto}\n")

                logging.info(f"Segmento {segment_id}: {formatar_timestamp(inicio)} --> {formatar_timestamp(fim)} {texto}")

            if fila_progresso:
                progress = segment_id / total_segments
                fila_progresso.put(progress)
            segment_id += 1

        # Fechar arquivos abertos
        for arquivo in arquivos.values():
            if hasattr(arquivo, 'close'):
                arquivo.close()

        logging.info("Arquivos de transcrição gerados.")
    except Exception as e:
        logging.error(f"Erro ao transcrever áudio para {caminho_audio}: {e}", exc_info=True)
        raise

def salvar_transcricao(segmentos, caminho_srt, caminho_fala_cronometrada):
    """Salva os segmentos transcritos em arquivos SRT e de Fala Cronometrada."""
    try:
        with open(caminho_srt, "w", encoding="utf-8") as arquivo_srt:
            with open(caminho_fala_cronometrada, "w", encoding="utf-8") as arquivo_txt:
                for segmento in segmentos:
                    inicio = segmento.start
                    fim = segmento.end
                    texto = segmento.text.strip()

                    logging.info(f"Salvando segmento {segmento.id} {formatar_timestamp(inicio)} --> {formatar_timestamp(fim)} {texto}")

                    # Arquivo SRT
                    arquivo_srt.write(f"{segmento.id}\n")
                    arquivo_srt.write(f"{formatar_timestamp(inicio)} --> {formatar_timestamp(fim)}\n")
                    arquivo_srt.write(f"{texto}\n\n")

                    # Arquivo de texto Fala Cronometrada
                    arquivo_txt.write(f"{formatar_timestamp(inicio)}: {texto}\n")
    except Exception as e:
        logging.error(f"Erro ao salvar transcrição: {e}", exc_info=True)
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
        transcrever_audio_faster_whisper(caminho_video, nome_modelo=nome_modelo, fila_progresso=fila_progresso)
        fila_progresso.put("Transcrição de áudio concluída!")
    except Exception as e:
        logging.error(f"Erro no processo de transcrição: {e}", exc_info=True)
        fila_progresso.put(f"Erro na transcrição: {e}")


def encontrar_arquivos_mascara(mascara, recursivo):
    """Encontra arquivos com base na máscara fornecida, mesmo em subpastas vazias."""
    try:
        if (recursivo):
            # Usar "**/" para garantir busca recursiva por todas as subpastas
            caminho_completo = os.path.join("**", mascara)  # Exemplo: '**/*.mp4'
            return glob.glob(caminho_completo, recursive=True)
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
    parser = argparse.ArgumentParser(description="Processa vídeos e áudios para extrair frames e gerar legendas com Faster-Whisper.")
    parser.add_argument("mascara_arquivos", type=str, nargs='?', help="Máscara de arquivos para processamento (ex: *.mp4, *.mp3).")
    parser.add_argument("--modelo", type=str, default="large-v3", help="Modelo Whisper a ser utilizado (padrão: small).")
    parser.add_argument("--recursivo", action="store_true", help="Busca recursivamente em subdiretórios.")
    parser.add_argument("--pasta_saida", type=str, help="Pasta de saída para armazenar frames.")
    parser.add_argument("--desativar-frames", action="store_true", help="Desativa o processamento de frames.")
    parser.add_argument("--skip-transcricao", action="store_true", help="Pula o processo de transcrição.")
    parser.add_argument("--skip-prontos", action="store_true", help="Pula arquivos já processados com '-Fala.Cronometrada.txt' maior que 1KB.")
    
    # Add new argument --no-interactive
    parser.add_argument("--no-interactive", action="store_true", help="Executa o programa sem a interface gráfica.")

    args = parser.parse_args()

    if not args.no_interactive:
        # If --no-interactive is not specified, show GUI to get parameters

        def submit_parameters():
            args.mascara_arquivos = entry_mascara_arquivos.get()
            args.modelo = entry_modelo.get()
            args.recursivo = var_recursivo.get()
            args.pasta_saida = entry_pasta_saida.get() if entry_pasta_saida.get() else None
            args.desativar_frames = var_desativar_frames.get()
            args.skip_prontos = var_skip_prontos.get()
            args.skip_transcricao = var_skip_transcricao.get()
            root.destroy()

        # Alterar o tema da janela principal
        root = ThemedTk(theme="radiance")  # Novo tema para a janela de configuração
        root.title("Configurações do Programa")
        style = ttk.Style(root)
        style.theme_use('radiance')  # Aplicar o novo tema

        # Input fields for parameters
        tk.Label(root, text="Máscara de Arquivos:").grid(row=0, column=0, sticky="e")
        entry_mascara_arquivos = tk.Entry(root, width=50)
        entry_mascara_arquivos.grid(row=0, column=1)
        entry_mascara_arquivos.insert(0, args.mascara_arquivos if args.mascara_arquivos else "*.mp4")

        tk.Label(root, text="Modelo Whisper:").grid(row=1, column=0, sticky="e")
        entry_modelo = tk.Entry(root, width=50)
        entry_modelo.grid(row=1, column=1)
        entry_modelo.insert(0, args.modelo)

        var_recursivo = tk.BooleanVar(value=args.recursivo)
        tk.Checkbutton(root, text="Busca Recursiva", variable=var_recursivo).grid(row=2, column=1, sticky="w")

        tk.Label(root, text="Pasta de Saída:").grid(row=3, column=0, sticky="e")
        entry_pasta_saida = tk.Entry(root, width=50)
        entry_pasta_saida.grid(row=3, column=1)
        if args.pasta_saida:
            entry_pasta_saida.insert(0, args.pasta_saida)

        var_desativar_frames = tk.BooleanVar(value=args.desativar_frames)
        tk.Checkbutton(root, text="Desativar Frames", variable=var_desativar_frames).grid(row=4, column=1, sticky="w")

        var_skip_prontos = tk.BooleanVar(value=args.skip_prontos)
        tk.Checkbutton(root, text="Pular Arquivos Prontos", variable=var_skip_prontos).grid(row=5, column=1, sticky="w")
        
        var_skip_transcricao = tk.BooleanVar(value=args.skip_transcricao if hasattr(args, 'skip_transcricao') else False)
        tk.Checkbutton(root, text="Pular Transcrição", variable=var_skip_transcricao).grid(row=6, column=1, sticky="w")
        
        # Ajuste o número da linha para o botão "Iniciar"
        tk.Button(root, text="Iniciar", command=submit_parameters).grid(row=7, column=1)

        root.mainloop()

    # Criar a janela de progresso com o novo tema
    progress_root = ThemedTk(theme="radiance")  # Novo tema para a janela de progresso
    progress_root.title("Progresso")
    style = ttk.Style(progress_root)
    style.theme_use('radiance')  # Aplicar o novo tema

    # Elementos da GUI
    tk.Label(progress_root, text="Arquivo Atual:").grid(row=0, column=0, sticky="w")
    arquivo_label = tk.Label(progress_root, text="", font=("Arial", 12, "bold"))
    arquivo_label.grid(row=0, column=1, sticky="w")

    tk.Label(progress_root, text="Processamento de Frames:").grid(row=1, column=0, sticky="w")
    progress_bar_frames = ttk.Progressbar(progress_root, length=400)
    progress_bar_frames.grid(row=1, column=1)

    tk.Label(progress_root, text="Transcrição de Áudio:").grid(row=2, column=0, sticky="w")
    progress_bar_transcricao = ttk.Progressbar(progress_root, length=400)
    progress_bar_transcricao.grid(row=2, column=1)

    tk.Label(progress_root, text="Logs:").grid(row=3, column=0, sticky="nw")
    log_label = tk.Label(progress_root, text="", justify="left", anchor="w")
    log_label.grid(row=3, column=1, sticky="w")

    # Custom logging handler to update GUI label
    class TkinterHandler(logging.Handler):
        def __init__(self, label):
            super().__init__()
            self.label = label

        def emit(self, record):
            msg = self.format(record)
            self.label.config(text=msg)
            self.label.update_idletasks()  # Adicione esta linha para atualizar a tela

    logger = logging.getLogger()

    # Do not remove existing handlers
    # Remove the loop that removes existing handlers
    # for handler in logger.handlers[:]:
    #     logger.removeHandler(handler)

    # Add the TkinterHandler to the logger
    logger.addHandler(TkinterHandler(log_label))

    # Set the logging level as needed
    logger.setLevel(logging.DEBUG)

    # Filas para monitorar progresso
    fila_progresso_transcricao = Queue()
    fila_progresso_frames = Queue()
    fila_progresso_arquivo = Queue()  # Nova fila para o arquivo atual

    # Função para atualizar a GUI periodicamente
    def update_gui():
        # Atualizar o nome do arquivo atual
        if not fila_progresso_arquivo.empty():
            current_file = fila_progresso_arquivo.get()
            arquivo_label.config(text=current_file)

        # Check progress queues
        if not fila_progresso_frames.empty():
            msg_frames = fila_progresso_frames.get()
            if isinstance(msg_frames, float):
                progress_bar_frames['value'] = msg_frames * 100
            else:
                log_label.config(text=msg_frames)

        if not fila_progresso_transcricao.empty():
            msg_transcricao = fila_progresso_transcricao.get()
            if isinstance(msg_transcricao, float):
                progress_bar_transcricao['value'] = msg_transcricao * 100
            else:
                log_label.config(text=msg_transcricao)

        # Schedule next check
        progress_root.after(100, update_gui)

    # Start updating the GUI
    update_gui()

    # Start processing in a separate thread to avoid blocking the GUI
    import threading
    processing_thread = threading.Thread(target=process_files, args=(args, fila_progresso_transcricao, fila_progresso_frames, fila_progresso_arquivo))
    processing_thread.start()

    # Start the main event loop
    progress_root.mainloop()

def process_files(args, fila_progresso_transcricao, fila_progresso_frames, fila_progresso_arquivo):
    # Verificar se GPU está disponível
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.debug(f"Usando dispositivo: {device}")

    if device == "cuda":
        # Resetar o dispositivo CUDA
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Usar a pasta do arquivo processado se pasta_saida não for fornecida
    for caminho_arquivo in encontrar_arquivos_mascara(args.mascara_arquivos, args.recursivo):
        try:
            # Enviar o nome do arquivo atual para a fila
            fila_progresso_arquivo.put(caminho_arquivo)

            # Verificar se o arquivo correspondente de fala cronometrada já existe e tem mais de 1KB
            caminho_fala_cronometrada = caminho_arquivo.replace(".mp4", "-Fala.Cronometrada.txt").replace(".mp3", "-Fala.Cronometrada.txt")

            if args.skip_prontos and os.path.exists(caminho_fala_cronometrada) and os.path.getsize(caminho_fala_cronometrada) > 1024:
                logging.info(f"Pulado: '{caminho_arquivo}', arquivo de fala cronometrada '{caminho_fala_cronometrada}' já existe e é maior que 1KB.")
                continue  # Pular para o próximo arquivo

            if args.pasta_saida:
                pasta_saida = args.pasta_saida
            else:
                pasta_saida = os.path.dirname(caminho_arquivo)  # mesma pasta do arquivo processado

            pasta_saida = os.path.join(pasta_saida, os.path.splitext(os.path.basename(caminho_arquivo))[0])
            os.makedirs(pasta_saida, exist_ok=True)

            logging.info(f"Processando arquivo: {caminho_arquivo}")
            logging.info(f"Pasta de saída: {pasta_saida}")

            if not args.skip_transcricao:
                # Criar e iniciar processo de transcrição
                processo_transcricao = Process(target=processar_transcricao, args=(caminho_arquivo, args.modelo, fila_progresso_transcricao))
                processo_transcricao.start()
            else:
                logging.info("Processo de transcrição pulado.")

            # Iniciar processo de frames apenas se não estiver desativado e for um vídeo
            if not args.desativar_frames and caminho_arquivo.endswith(".mp4"):
                processo_frames = Process(target=processar_frames, args=(caminho_arquivo, pasta_saida, fila_progresso_frames))
                processo_frames.start()
            else:
                logging.info("Processamento de frames desativado ou não aplicável.")

            # Monitorar progresso
            while ((not args.skip_transcricao and processo_transcricao.is_alive()) or
                   (not args.desativar_frames and caminho_arquivo.endswith(".mp4") and processo_frames.is_alive())):
                if not args.desativar_frames and caminho_arquivo.endswith(".mp4") and not fila_progresso_frames.empty():
                    msg_frames = fila_progresso_frames.get()
                    logging.info(f"Frames: {msg_frames}")

                if not fila_progresso_transcricao.empty():
                    msg_transcricao = fila_progresso_transcricao.get()
                    logging.info(f"Transcrição: {msg_transcricao}")

            # Garantir que ambos os processos foram concluídos
            if not args.skip_transcricao:
                processo_transcricao.join()
            if not args.desativar_frames and caminho_arquivo.endswith(".mp4"):
                processo_frames.join()

        except Exception as e:
            logging.error(f"Erro ao processar arquivo {caminho_arquivo}: {e}")

    tempo_total = time() - tempo_inicio
    tempo_formatado = formatar_tempo_humano(tempo_total)
    logging.info(f"Processo concluído em {tempo_formatado}.")

if __name__ == "__main__":
    tempo_inicio = time()  # Capturar o tempo de início antes da execução principal
    main()
