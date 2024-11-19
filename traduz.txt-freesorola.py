import argparse
import os
import glob
from translate import Translator
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def is_metadata(line):
    """Verifica se a linha é metadado (timestamps ou outros)."""
    return line.strip().startswith("NOTE") or "-->" in line or line.strip() == ""

def translate_line(line, target_lang):
    """Traduz uma única linha usando a biblioteca translate."""
    translator = Translator(to_lang=target_lang)
    try:
        if line.strip() and not is_metadata(line):  # Traduz apenas conteúdo falado
            return translator.translate(line.strip())
        else:
            return line  # Retorna a linha original se for metadado
    except Exception as e:
        print(f"Erro ao traduzir a linha: {line}. Erro: {e}")
        return line  # Retorna a linha original em caso de erro

def translate_file(input_file, output_file, target_lang, processes):
    """Traduz o conteúdo de um arquivo de texto ou `.vtt` e salva em outro arquivo."""
    try:
        print(f"Processando o arquivo: {input_file}")
        
        # Lê o conteúdo do arquivo de entrada
        with open(input_file, 'r', encoding='utf-8') as infile:
            content = infile.readlines()

        # Traduz as linhas em paralelo com uma barra de progresso
        translated_lines = []
        with ThreadPoolExecutor(max_workers=processes) as executor:
            futures = {executor.submit(translate_line, line, target_lang): idx for idx, line in enumerate(content)}

            with tqdm(total=len(futures), desc=f"Traduzindo {input_file}", unit="linha") as progress:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        translated_lines.append((idx, future.result()))
                    except Exception as e:
                        print(f"Erro ao processar a linha {idx}: {e}")
                    progress.update(1)

        # Ordena as linhas traduzidas para preservar a ordem original
        translated_lines.sort(key=lambda x: x[0])
        translated_content = "\n".join([line for _, line in translated_lines])

        # Salva o texto traduzido no arquivo de saída
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(translated_content)
        
        print(f"Tradução do arquivo '{input_file}' concluída. Salvo em: {output_file}\n")
    except Exception as e:
        print(f"Erro ao processar o arquivo '{input_file}': {e}\n")

def process_files(input_mask, target_lang, processes):
    """Processa todos os arquivos que correspondem à máscara de entrada."""
    files = glob.glob(input_mask)
    if not files:
        print("Nenhum arquivo encontrado para a máscara especificada.")
        return

    print(f"Encontrados {len(files)} arquivos para tradução.\n")
    for input_file in files:
        # Gera o nome do arquivo de saída com o sufixo -pt.BR
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}-pt.BR{ext}"
        translate_file(input_file, output_file, target_lang, processes)

    print("Processamento de todos os arquivos concluído!")

def main():
    parser = argparse.ArgumentParser(description="Traduz texto de arquivos ou conteúdo falado em arquivos VTT.")
    parser.add_argument('input_mask', help="Máscara para os arquivos de entrada (ex: 'textos/*.vtt').")
    parser.add_argument('--target-lang', default='pt', help="Idioma de destino (padrão: pt).")
    parser.add_argument('--processes', type=int, default=8, help="Número de threads paralelas por arquivo (padrão: 8).")
    args = parser.parse_args()

    process_files(args.input_mask, args.target_lang, args.processes)

if __name__ == "__main__":
    main()
