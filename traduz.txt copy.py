import argparse
import os
import glob
from googletrans import Translator

def translate_line_by_line(content, target_lang):
    """Traduz o texto linha por linha para evitar problemas de limite."""
    translator = Translator()
    translated_lines = []
    for line in content.splitlines():
        try:
            if line.strip():  # Ignorar linhas vazias
                translated = translator.translate(line, dest=target_lang)
                translated_lines.append(translated.text)
            else:
                translated_lines.append("")
        except Exception as e:
            print(f"Erro ao traduzir a linha: {line}. Erro: {e}")
            translated_lines.append(line)  # Mantém a linha original em caso de falha
    return "\n".join(translated_lines)

def translate_file(input_file, output_file, target_lang):
    """Traduz o conteúdo de um arquivo de texto e salva em outro arquivo."""
    try:
        # Lê o conteúdo do arquivo de entrada
        with open(input_file, 'r', encoding='utf-8') as infile:
            content = infile.read()
        
        # Traduz o conteúdo linha por linha
        translated_content = translate_line_by_line(content, target_lang)
        
        # Salva o texto traduzido no arquivo de saída
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(translated_content)
        
        print(f"Tradução concluída. Arquivo salvo em: {output_file}")
    except Exception as e:
        print(f"Erro ao processar o arquivo '{input_file}': {e}")

def process_files(input_mask, target_lang):
    """Processa todos os arquivos que correspondem à máscara de entrada."""
    files = glob.glob(input_mask)
    if not files:
        print("Nenhum arquivo encontrado para a máscara especificada.")
        return

    for input_file in files:
        # Gera o nome do arquivo de saída com o sufixo -pt.BR
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}-pt.BR{ext}"
        translate_file(input_file, output_file, target_lang)

def main():
    parser = argparse.ArgumentParser(description="Traduz texto de arquivos usando o Google Translator.")
    parser.add_argument('input_mask', help="Máscara para os arquivos de entrada (ex: 'textos/*.txt').")
    parser.add_argument('--target-lang', default='pt', help="Idioma de destino (padrão: pt).")
    args = parser.parse_args()

    process_files(args.input_mask, args.target_lang)

if __name__ == "__main__":
    main()
