# Processador de Vídeo para CSV

Este projeto extrai texto de vídeos usando OCR e salva os resultados em um arquivo CSV.

## Pré-requisitos

1. Python 3.7 ou superior
2. Tesseract-OCR instalado no sistema

### Instalação do Tesseract-OCR

#### Windows:
1. Baixe o instalador do Tesseract-OCR em: https://github.com/UB-Mannheim/tesseract/wiki
2. Instale e adicione ao PATH do sistema

#### Linux:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

## Instalação

1. Clone este repositório
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

1. Coloque seu vídeo no diretório do projeto
2. Modifique o arquivo `video_to_csv.py` para apontar para seu vídeo
3. Execute o script:
```bash
python video_to_csv.py
```

O script irá:
1. Extrair frames do vídeo
2. Processar cada frame com OCR
3. Salvar os resultados em um arquivo CSV

## Configurações

- `frame_interval`: Controla quantos frames são processados (1 = todos os frames, 30 = 1 frame por segundo em vídeos de 30fps)
- Os frames extraídos são salvos no diretório `frames/`
- O resultado final é salvo em `resultados.csv` 