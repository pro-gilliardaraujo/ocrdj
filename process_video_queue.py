import cv2
import pytesseract
import numpy as np
import os
from pathlib import Path
import time
import json
import shutil

# Configuração do Tesseract
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR'
if TESSERACT_PATH not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + TESSERACT_PATH
pytesseract.pytesseract.tesseract_cmd = os.path.join(TESSERACT_PATH, 'tesseract.exe')

def enhance_image_for_ocr(image, roi_type):
    """
    Função otimizada para pré-processamento de imagem usando apenas o método mais eficiente para cada tipo
    """
    if image is None or image.size == 0:
        return None

    # Converte para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Redimensiona a imagem (aumenta o tamanho para melhor reconhecimento)
    scale = 4 if roi_type == 'spacing' else 3 if roi_type == 'speed' else 2
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    preprocessed_images = []
    
    # Para todos os tipos, usa apenas Normalização básica
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    preprocessed_images.append(("Normalização básica", norm))
    
    return preprocessed_images

def extract_value_basic(roi_img):
    # Pré-processamento básico
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Configuração do OCR
    config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.'
    
    # Extração do texto
    text = pytesseract.image_to_string(thresh, config=config).strip()
    text = ''.join(c for c in text if c.isdigit() or c == '.')
    
    confidence = 'high' if text else 'low'
    return text, confidence, thresh

def extract_value_with_debug(roi_img, roi_type):
    debug_info = {}
    final_value = {'value': None, 'method': None, 'confidence': None}
    
    # Try basic normalization
    try:
        print(f"\nProcessando {roi_type} com Normalização básica:")
        value, confidence, processed_img = extract_value_basic(roi_img)
        print(f"Texto extraído: {value}")
        print(f"Texto original: {value}")
        
        if value:
            initial_value = float(value)
            print(f"Valor inicial: {initial_value}")
            
            # Adjust spray rate if needed
            if roi_type == 'spray_rate' and initial_value > 100:
                initial_value = initial_value / 100
                print(f"spray_rate > 100, ajustando para: {initial_value:.2f}")
            
            # Validate height
            if roi_type == 'height':
                if 10 <= initial_value <= 20:
                    print(f"Altura entre 10-20, mantendo: {initial_value}")
                else:
                    print(f"Altura fora do intervalo 10-20: {initial_value}")
                    confidence = 'low'
            
            print(f"Valor válido: True, Confiança: {confidence}")
            
            debug_info['Normalização básica'] = {
                'value': initial_value,
                'confidence': confidence,
                'image': processed_img
            }
            
            final_value = {
                'value': initial_value,
                'method': 'Normalização básica',
                'confidence': confidence
            }
    except Exception as e:
        print(f"Erro na normalização básica: {str(e)}")
    
    # Print final result
    if final_value['value'] is not None:
        print(f"\nResultado final para {roi_type}:")
        print(f"Valor: {final_value['value']}")
        print(f"Método: {final_value['method']}")
        print(f"Confiança: {final_value['confidence']}")
    
    return final_value, debug_info

def process_frame(frame, frame_num, debug_dir, rois):
    """
    Processa um frame do vídeo e retorna os resultados e informações de debug
    """
    frame_dir = os.path.join(debug_dir, f"frame_{frame_num}")
    os.makedirs(frame_dir, exist_ok=True)
    
    # Salva o frame completo
    cv2.imwrite(os.path.join(frame_dir, "frame_completo.png"), frame)
    
    results = {}
    debug_info = {}
    
    h, w = frame.shape[:2]
    
    # Processa cada ROI
    for roi_type, roi_info in rois.items():
        print(f"\nTestando ROI: {roi_type}")
        
        # Extrai a ROI
        x1 = int(w * roi_info['x'])
        y1 = int(h * roi_info['y'])
        x2 = int(w * (roi_info['x'] + roi_info['width']))
        y2 = int(h * (roi_info['y'] + roi_info['height']))
        roi_img = frame[y1:y2, x1:x2]
        
        # Cria diretório para as imagens da ROI
        roi_dir = os.path.join(frame_dir, roi_type)
        os.makedirs(roi_dir, exist_ok=True)
        
        # Processa a ROI
        value, roi_debug = extract_value_with_debug(roi_img, roi_type)
        
        # Salva as imagens processadas
        for method, result in roi_debug.items():
            if 'image' in result:
                method_filename = method.lower().replace(' ', '_').replace('ç', 'c').replace('ã', 'a')
                img_path = os.path.join(roi_dir, f"{method_filename}.png")
                cv2.imwrite(img_path, result['image'])
        
        results[roi_type] = value
        debug_info[roi_type] = roi_debug
    
    return results, debug_info

def normalize_frame_size(frame, target_width=1920, target_height=1080):
    """
    Normaliza o tamanho do frame para as dimensões padrão
    """
    if frame is None:
        return None
        
    current_height, current_width = frame.shape[:2]
    
    # Se as dimensões já são as desejadas, retorna o frame original
    if current_width == target_width and current_height == target_height:
        return frame
    
    # Calcula a proporção atual
    current_ratio = current_width / current_height
    target_ratio = target_width / target_height
    
    # Determina as dimensões finais mantendo a proporção
    if current_ratio > target_ratio:
        # Imagem mais larga que o padrão
        new_width = target_width
        new_height = int(target_width / current_ratio)
    else:
        # Imagem mais alta que o padrão
        new_height = target_height
        new_width = int(target_height * current_ratio)
    
    # Redimensiona a imagem
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Cria uma imagem preta do tamanho padrão
    normalized = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Calcula as posições para centralizar a imagem
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    # Coloca a imagem redimensionada no centro
    normalized[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return normalized

def process_video(video_path, rois, output_dir):
    """
    Processa um vídeo e retorna os resultados
    """
    # Abre o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return None
    
    # Obtém informações do vídeo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n\nIniciando processamento do vídeo: {video_path}")
    print(f"FPS: {fps}")
    print(f"Dimensões originais: {width}x{height}")
    print(f"Total de frames: {total_frames}\n")
    
    # Cria diretório temporário para os resultados deste vídeo
    video_name = Path(video_path).stem
    video_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_dir, exist_ok=True)
    
    # Lista para armazenar resultados
    video_results = []
    
    # Calcula o intervalo de frames para 2 FPS
    frame_interval = fps // 2  # 2 frames por segundo
    
    for frame_num in range(0, total_frames, frame_interval):
        print(f"\nProcessando frame {frame_num} ({frame_num/total_frames*100:.1f}%)")
        
        # Define a posição do frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Erro ao ler o frame {frame_num}")
            continue
        
        # Normaliza o tamanho do frame
        frame = normalize_frame_size(frame)
        
        # Processa o frame
        results, _ = process_frame(frame, frame_num, video_dir, rois)
        
        # Armazena os resultados
        frame_results = {
            'Video': video_name,
            'Frame': frame_num,
            'Tempo (s)': frame_num / fps,
            'Taxa de Aplicação (L/min)': results.get('spray_rate', {}).get('value', ''),
            'Altura (m)': results.get('height', {}).get('value', ''),
            'Espaçamento (m)': results.get('spacing', {}).get('value', ''),
            'Velocidade (m/s)': results.get('speed', {}).get('value', '')
        }
        video_results.append(frame_results)
    
    cap.release()
    
    # Remove diretório temporário com frames e recortes
    shutil.rmtree(video_dir)
    
    return video_results

def move_to_processed(video_path, processed_dir):
    """
    Move o vídeo processado para a pasta de processados
    """
    video_name = os.path.basename(video_path)
    dest_path = os.path.join(processed_dir, video_name)
    
    # Se já existe um arquivo com o mesmo nome, adiciona timestamp
    if os.path.exists(dest_path):
        base, ext = os.path.splitext(video_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dest_path = os.path.join(processed_dir, f"{base}_{timestamp}{ext}")
    
    shutil.move(video_path, dest_path)
    print(f"Vídeo movido para: {dest_path}")

def main():
    # Diretórios de trabalho
    queue_dir = "fila"
    processed_dir = "processados"
    results_dir = "resultados"
    
    # Cria os diretórios se não existirem
    os.makedirs(queue_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Lista todos os vídeos na fila
    video_extensions = ('.mp4', '.avi', '.mkv')
    videos = [f for f in os.listdir(queue_dir) 
             if os.path.splitext(f)[1].lower() in video_extensions]
    
    if not videos:
        print("Nenhum vídeo encontrado na pasta 'fila'")
        return
    
    # Carrega ROIs salvas
    if not os.path.exists('last_rois.json'):
        print("Arquivo de ROIs não encontrado. Por favor, defina as ROIs primeiro no programa principal.")
        return
    
    with open('last_rois.json', 'r') as f:
        rois = json.load(f)
    
    if not any(rois.values()):
        print("Nenhuma ROI definida. Por favor, defina as ROIs primeiro no programa principal.")
        return
    
    # Processa cada vídeo na fila
    timestamp = time.strftime("%d%m%Y_%H%M%S")
    all_results = []
    
    print(f"\nIniciando processamento de {len(videos)} vídeos...")
    
    for video_name in videos:
        video_path = os.path.join(queue_dir, video_name)
        print(f"\nProcessando: {video_name}")
        
        try:
            # Processa o vídeo
            video_results = process_video(video_path, rois.copy(), results_dir)
            
            if video_results:
                all_results.extend(video_results)
                # Move o vídeo para a pasta de processados
                move_to_processed(video_path, processed_dir)
            else:
                print(f"Erro ao processar o vídeo: {video_name}")
        
        except Exception as e:
            print(f"Erro ao processar o vídeo {video_name}: {str(e)}")
            continue
    
    # Salva todos os resultados em um único CSV
    if all_results:
        csv_path = os.path.join(results_dir, f"resultados_{timestamp}.csv")
        print(f"\nSalvando todos os resultados em: {csv_path}")
        
        try:
            with open(csv_path, "w", encoding="utf-8") as f:
                # Escreve o cabeçalho
                headers = ['Video', 'Frame', 'Tempo (s)', 'Taxa de Aplicação (L/min)', 
                         'Altura (m)', 'Espaçamento (m)', 'Velocidade (m/s)']
                f.write(';'.join(headers) + '\n')
                
                # Escreve os dados
                for result in all_results:
                    row = [str(result[header]).replace('.', ',') for header in headers]
                    f.write(';'.join(row) + '\n')
            
            print(f"CSV gerado com sucesso! Total de registros: {len(all_results)}")
        except Exception as e:
            print(f"ERRO ao salvar CSV: {str(e)}")
    
    print("\nProcessamento em lote concluído!")

if __name__ == "__main__":
    main() 