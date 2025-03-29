import cv2
import pytesseract
import numpy as np
import os
from pathlib import Path
import time
import json

# Configuração do Tesseract
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR'
if TESSERACT_PATH not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + TESSERACT_PATH
pytesseract.pytesseract.tesseract_cmd = os.path.join(TESSERACT_PATH, 'tesseract.exe')

def enhance_image_for_ocr(image, roi_type):
    """
    Função otimizada para pré-processamento de imagem usando vários métodos
    """
    if image is None or image.size == 0:
        return None

    # Converte para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aumenta significativamente o fator de escala, especialmente para espaçamento e altura
    scale = 8 if roi_type in ['spacing', 'height'] else 7 if roi_type == 'speed' else 6
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    preprocessed_images = []
    
    # Para altura e espaçamento, aplica técnicas específicas
    if roi_type in ['spacing', 'height']:
        # 1. Threshold adaptativo com kernel maior
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        adaptive1 = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
        preprocessed_images.append(("Adaptativo Gaussiano", adaptive1))
        
        # 2. Threshold Otsu com blur mais forte
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("Otsu", otsu))
        
        # 3. CLAHE com parâmetros ajustados
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        cl1 = clahe.apply(gray)
        _, thresh_eq = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("CLAHE", thresh_eq))
        
        # 4. Dilatação seguida de erosão para melhorar a conectividade dos dígitos
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(gray, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        _, morph = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("Morfológico", morph))
    else:
        # Para outros tipos de ROI, mantém o processamento original
        # 1. Normalização básica com threshold adaptativo
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        adaptive1 = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
        preprocessed_images.append(("Adaptativo Gaussiano", adaptive1))
        
        # 2. Threshold Otsu com blur
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("Otsu", otsu))
        
        # 3. Threshold adaptativo direto
        adaptive2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)
        preprocessed_images.append(("Adaptativo Médio", adaptive2))
        
        # 4. CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(gray)
        _, thresh_eq = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("CLAHE", thresh_eq))
    
    return preprocessed_images

def extract_value_from_image(roi_img, roi_type):
    """
    Extrai o valor da ROI usando diferentes métodos de pré-processamento
    """
    # Obtém as imagens pré-processadas
    preprocessed_images = enhance_image_for_ocr(roi_img, roi_type)
    if not preprocessed_images:
        return None, 'low', None
    
    best_result = {'value': None, 'confidence': 'low', 'image': None, 'method': None}
    
    # Configuração do OCR
    config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.'
    
    # Tenta cada método de pré-processamento
    for method_name, processed_img in preprocessed_images:
        try:
            # Extração do texto
            text = pytesseract.image_to_string(processed_img, config=config).strip()
            text = ''.join(c for c in text if c.isdigit() or c == '.')
            
            if text:
                try:
                    value = float(text)
                    
                    # Ajusta valores muito grandes
                    if '.' not in text:  # Se não tem ponto decimal
                        if roi_type == 'height':
                            # Para altura, se o valor for maior que 20, divide por 10
                            if value > 20:
                                value = value / 10
                        elif value > 100:
                            value = value / 100
                        elif value > 10:
                            value = value / 10
                    
                    # Verifica se o valor está dentro dos limites esperados
                    is_valid = False
                    if roi_type == 'spray_rate' and 0.1 <= value <= 10:
                        is_valid = True
                    elif roi_type == 'height' and 0.1 <= value <= 20:
                        is_valid = True
                    elif roi_type == 'spacing' and 0.1 <= value <= 5:
                        is_valid = True
                    elif roi_type == 'speed' and 0.1 <= value <= 20:
                        is_valid = True
                    
                    if is_valid:
                        confidence = 'high'
                        # Se encontrou um valor válido com alta confiança, usa este
                        if best_result['confidence'] != 'high' or best_result['value'] is None:
                            best_result = {
                                'value': value,
                                'confidence': confidence,
                                'image': processed_img,
                                'method': method_name
                            }
                    else:
                        # Se o valor atual não é válido mas ainda não temos nenhum resultado
                        if best_result['value'] is None:
                            best_result = {
                                'value': value,
                                'confidence': 'low',
                                'image': processed_img,
                                'method': method_name
                            }
                
                except ValueError:
                    continue
        
        except Exception as e:
            print(f"Erro ao processar imagem com {method_name}: {str(e)}")
            continue
    
    return best_result['value'], best_result['confidence'], best_result['image']

def extract_value_basic(roi_img):
    """
    Mantida para compatibilidade, usa o novo método de extração
    """
    return extract_value_from_image(roi_img, 'spray_rate')

def extract_value_otsu(roi_img):
    """
    Mantida para compatibilidade, usa o novo método de extração
    """
    return extract_value_from_image(roi_img, 'spacing')

def extract_value_with_debug(roi_img, roi_type):
    """
    Extrai o valor da ROI com informações de debug
    """
    debug_info = {}
    final_value = {'value': None, 'method': None, 'confidence': None}
    
    try:
        # Obtém as imagens pré-processadas
        preprocessed_images = enhance_image_for_ocr(roi_img, roi_type)
        if not preprocessed_images:
            print(f"\nErro: Não foi possível pré-processar a imagem para {roi_type}")
            return final_value, debug_info
        
        print(f"\nProcessando {roi_type}:")
        best_result = {'value': None, 'confidence': 'low', 'image': None, 'method': None}
        
        # Configuração do OCR
        config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.'
        
        # Tenta cada método de pré-processamento
        for method_name, processed_img in preprocessed_images:
            try:
                print(f"\nTentando método: {method_name}")
                
                # Extração do texto
                text = pytesseract.image_to_string(processed_img, config=config).strip()
                text = ''.join(c for c in text if c.isdigit() or c == '.')
                print(f"Texto extraído: {text}")
                
                if text:
                    try:
                        value = float(text)
                        print(f"Valor numérico: {value}")
                        
                        # Ajusta valores muito grandes
                        if value > 100:
                            value = value / 100
                            print(f"Valor ajustado (>100): {value:.2f}")
                        
                        # Verifica se o valor está dentro dos limites esperados
                        is_valid = False
                        if roi_type == 'spray_rate' and 0.1 <= value <= 10:
                            is_valid = True
                        elif roi_type == 'height' and 0.1 <= value <= 20:
                            is_valid = True
                        elif roi_type == 'spacing' and 0.1 <= value <= 5:
                            is_valid = True
                        elif roi_type == 'speed' and 0.1 <= value <= 20:
                            is_valid = True
                        
                        confidence = 'high' if is_valid else 'low'
                        print(f"Valor válido: {is_valid}, Confiança: {confidence}")
                        
                        # Armazena o resultado deste método
                        debug_info[method_name] = {
                            'value': value,
                            'confidence': confidence,
                            'image': processed_img
                        }
                        
                        # Atualiza o melhor resultado se necessário
                        if is_valid and (best_result['confidence'] != 'high' or best_result['value'] is None):
                            best_result = {
                                'value': value,
                                'confidence': confidence,
                                'image': processed_img,
                                'method': method_name
                            }
                        elif not is_valid and best_result['value'] is None:
                            best_result = {
                                'value': value,
                                'confidence': confidence,
                                'image': processed_img,
                                'method': method_name
                            }
                    
                    except ValueError:
                        print(f"Erro ao converter texto para número: {text}")
                        continue
            
            except Exception as e:
                print(f"Erro ao processar imagem com {method_name}: {str(e)}")
                continue
        
        # Define o resultado final
        if best_result['value'] is not None:
            final_value = {
                'value': best_result['value'],
                'method': best_result['method'],
                'confidence': best_result['confidence']
            }
    
    except Exception as e:
        print(f"Erro no processamento: {str(e)}")
    
    # Print final result
    if final_value['value'] is not None:
        print(f"\nResultado final para {roi_type}:")
        print(f"Valor: {final_value['value']}")
        print(f"Método: {final_value['method']}")
        print(f"Confiança: {final_value['confidence']}")
    
    return final_value, debug_info

def save_roi_image(roi_img, frame_dir, roi_type, method):
    # Normalize method name for file path
    method_filename = method.lower().replace(' ', '_').replace('ç', 'c').replace('ã', 'a')
    img_path = os.path.join(frame_dir, roi_type, f"{method_filename}.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    cv2.imwrite(img_path, roi_img)
    return img_path

def generate_html_report(results_dir, video_results):
    """
    Gera um relatório HTML com o formato original mais limpo
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Relatório de Processamento de Vídeo</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0;
                padding: 20px;
                background-color: #f0f0f0;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 { 
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #2c3e50;
                margin-top: 30px;
            }
            .frame-info {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 4px;
                margin: 20px 0;
            }
            .frame-image {
                text-align: center;
                margin: 20px 0;
            }
            .frame-image img {
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            .results-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background-color: white;
            }
            .results-table th, .results-table td {
                padding: 12px;
                border: 1px solid #ddd;
                text-align: left;
            }
            .results-table th {
                background-color: #3498db;
                color: white;
            }
            .results-table tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .value {
                font-family: monospace;
                background-color: #f8f9fa;
                padding: 2px 6px;
                border-radius: 3px;
            }
            .high {
                color: #27ae60;
                font-weight: bold;
            }
            .low {
                color: #e74c3c;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Relatório de Processamento de Vídeo</h1>
    """
    
    for video_name, frames in video_results.items():
        html_content += f"<h2>Vídeo: {video_name}</h2>"
        
        for frame_num, frame_data in frames.items():
            html_content += f"""
            <div class="frame-info">
                <h3>Frame {frame_num}</h3>
                <div class="frame-image">
                    <img src="frames/frame_{frame_num}/frame_completo.png" alt="Frame completo">
                </div>
                
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Região</th>
                            <th>Valor Detectado</th>
                            <th>Confiança</th>
                            <th>Imagem Original</th>
                            <th>Imagem Processada</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            roi_names = {
                'spray_rate': 'Taxa de Aplicação (L/min)',
                'height': 'Altura (m)',
                'spacing': 'Espaçamento (m)',
                'speed': 'Velocidade (m/s)'
            }
            
            for roi_type, roi_data in frame_data.get('debug_info', {}).items():
                best_result = frame_data['results'].get(roi_type, {})
                best_value = best_result.get('value', 'N/A')
                confidence = best_result.get('confidence', 'low')
                
                # Pega a primeira técnica disponível para mostrar as imagens
                first_technique = next(iter(roi_data))
                first_data = roi_data[first_technique]
                
                html_content += f"""
                <tr>
                    <td>{roi_names.get(roi_type, roi_type)}</td>
                    <td class="value">{best_value:.2f if isinstance(best_value, float) else best_value}</td>
                    <td class="{confidence}">{confidence.upper()}</td>
                    <td>
                        <img src="frames/frame_{frame_num}/{roi_type}/original.png" 
                             alt="Original {roi_type}" style="max-height: 100px;">
                    </td>
                    <td>
                        <img src="frames/frame_{frame_num}/{roi_type}/otsu.png" 
                             alt="Processada {roi_type}" style="max-height: 100px;">
                    </td>
                </tr>
                """
            
            html_content += """
                    </tbody>
                </table>
            </div>
            """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    return html_content

def save_test_history(results_data, timestamp, video_path, rois):
    """
    Salva os resultados do teste em um arquivo JSON para histórico
    """
    history_dir = Path("historico_testes")
    history_dir.mkdir(exist_ok=True)
    
    # Prepara os dados do teste
    test_data = {
        'timestamp': timestamp,
        'video_path': str(video_path),
        'rois': rois,
        'frames_analyzed': len(results_data),
        'results_by_frame': {}
    }
    
    # Organiza os resultados por frame
    for frame_data in results_data:
        frame_number = frame_data['frame_number']
        test_data['results_by_frame'][frame_number] = {
            'percent': frame_data['frame_percent'],
            'final_results': frame_data['final_results']
        }
    
    # Calcula estatísticas gerais
    stats = {roi_type: {'success_rate': 0, 'avg_confidence': 0} 
            for roi_type in ['spray_rate', 'height', 'spacing', 'speed']}
    
    for frame_data in results_data:
        for roi_type in stats:
            if roi_type in frame_data['final_results']:
                result = frame_data['final_results'][roi_type]
                if result.get('value') is not None:
                    stats[roi_type]['success_rate'] += 1
                    stats[roi_type]['avg_confidence'] += 1 if result['confidence'] == 'high' else 0.5
    
    # Calcula médias
    for roi_type in stats:
        total_frames = len(results_data)
        stats[roi_type]['success_rate'] = (stats[roi_type]['success_rate'] / total_frames) * 100
        stats[roi_type]['avg_confidence'] = (stats[roi_type]['avg_confidence'] / total_frames) * 100
    
    test_data['statistics'] = stats
    
    # Salva o arquivo JSON
    history_file = history_dir / f"teste_{timestamp}.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    
    return history_file

def adjust_spacing_roi(rois):
    """
    Ajusta a ROI do spacing movendo para a posição ideal
    """
    if 'spacing' in rois and rois['spacing']:
        # Mantém posição X e largura atuais
        rois['spacing']['x'] = 0.555
        rois['spacing']['width'] = 0.035
        
        # Ajusta a posição Y movendo 0.012 (1.2 pixels em coordenadas normalizadas) para baixo
        if 'y' in rois['spacing']:
            rois['spacing']['y'] += 0.012
        
        print("\nROI do spacing ajustada:")
        print(f"Nova posição X: {rois['spacing']['x']:.3f}")
        print(f"Nova posição Y: {rois['spacing']['y']:.3f}")
        print(f"Nova largura: {rois['spacing']['width']:.3f}\n")
    return rois

def process_frame(frame, frame_num, rois):
    """
    Processa um frame do vídeo usando as ROIs definidas pelo usuário
    """
    if not rois:
        print("Erro: Nenhuma ROI definida")
        return {}
    
    results = {}
    h, w = frame.shape[:2]
    
    # Processa cada ROI definida pelo usuário
    for roi_type, roi_info in rois.items():
        if not roi_info:  # Pula se a ROI não foi definida
            continue
            
        # Calcula as coordenadas da ROI usando as definidas pelo usuário
        x1 = int(w * roi_info['x'])
        y1 = int(h * roi_info['y'])
        x2 = int(w * (roi_info['x'] + roi_info['width']))
        y2 = int(h * (roi_info['y'] + roi_info['height']))
        
        # Extrai a região da ROI
        roi_img = frame[y1:y2, x1:x2]
        
        # Processa com diferentes técnicas
        preprocessed_images = enhance_image_for_ocr(roi_img, roi_type)
        if not preprocessed_images:
            continue
        
        best_value = None
        best_confidence = 'low'
        
        for technique, processed_img in preprocessed_images:
            try:
                # Configuração do OCR apenas para dígitos e ponto
                config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.'
                
                # Extrai o texto
                text = pytesseract.image_to_string(processed_img, config=config).strip()
                text = ''.join(c for c in text if c.isdigit() or c == '.')
                
                if text:
                    try:
                        value = float(text)
                        
                        # Ajusta valores para o espaçamento
                        if roi_type == 'spacing':
                            # Se o valor for maior que 5, assume que é um decimal sem ponto
                            if value > 5:
                                value = value / 10
                            # Validação mais flexível para espaçamento
                            is_valid = 0.1 <= value <= 5
                        else:
                            # Para outros tipos, mantém a lógica original
                            if value > 100:
                                value /= 100
                            
                            # Validações específicas por tipo
                            is_valid = False
                            if roi_type == 'spray_rate' and 0.1 <= value <= 10:
                                is_valid = True
                            elif roi_type == 'height' and 0.1 <= value <= 20:
                                is_valid = True
                            elif roi_type == 'speed' and 0.1 <= value <= 20:
                                is_valid = True
                        
                        confidence = 'high' if is_valid else 'low'
                        
                        # Atualiza o melhor resultado se necessário
                        if is_valid and (best_value is None or confidence == 'high'):
                            best_value = value
                            best_confidence = confidence
                    
                    except ValueError:
                        continue
            
            except Exception as e:
                continue
        
        if best_value is not None:
            results[roi_type] = {
                'value': best_value,
                'confidence': best_confidence
            }
    
    return results

def test_roi_extraction(video_path, rois):
    """
    Função principal de teste que processa o vídeo e extrai os valores das ROIs
    """
    # Abre o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return None
    
    # Obtém informações do vídeo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    print(f"\nProcessando vídeo: {video_name}")
    print(f"FPS: {fps}")
    print(f"Total de frames: {total_frames}")
    print(f"Duração aproximada: {total_frames/fps:.2f} segundos")
    
    # Lista para armazenar todos os resultados
    all_results = []
    
    # Calcula o intervalo de frames para 1 FPS
    frame_interval = fps  # 1 frame por segundo
    
    # Processa cada frame do vídeo no intervalo definido
    for frame_num in range(0, total_frames, frame_interval):
        # Atualiza o progresso
        percent = (frame_num/total_frames)*100
        print(f"\rProcessando frame {frame_num}/{total_frames} ({percent:.1f}%) - {frame_num//fps}s", end='')
        
        # Lê o frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"\nErro ao ler o frame {frame_num}")
            continue
        
        # Processa o frame
        results = process_frame(frame, frame_num, rois)
        
        # Armazena os resultados
        frame_data = {
            'frame_number': frame_num,
            'frame_time': frame_num/fps,
            'frame_percent': (frame_num/total_frames)*100,
            'final_results': results
        }
        all_results.append(frame_data)
    
    cap.release()
    print("\nProcessamento dos frames concluído!")
    
    # Salva o arquivo CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f"{video_name}_resultados_{timestamp}.csv"
    
    with open(csv_path, "w", encoding="utf-8") as f:
        # Escreve o cabeçalho
        headers = ['Frame', 'Tempo (s)', 'Taxa de Aplicação (L/min)', 'Altura (m)', 'Espaçamento (m)', 'Velocidade (m/s)']
        f.write(';'.join(headers) + '\n')
        
        # Escreve os dados de cada frame
        for frame_data in all_results:
            frame_num = frame_data['frame_number']
            frame_time = frame_data['frame_time']
            results = frame_data['final_results']
            
            row = [
                str(frame_num),
                f"{frame_time:.2f}".replace('.', ','),
                str(results.get('spray_rate', {}).get('value', '')).replace('.', ',') if results.get('spray_rate', {}).get('value') is not None else '',
                str(results.get('height', {}).get('value', '')).replace('.', ',') if results.get('height', {}).get('value') is not None else '',
                str(results.get('spacing', {}).get('value', '')).replace('.', ',') if results.get('spacing', {}).get('value') is not None else '',
                str(results.get('speed', {}).get('value', '')).replace('.', ',') if results.get('speed', {}).get('value') is not None else ''
            ]
            f.write(';'.join(row) + '\n')
    
    print(f"\nArquivo CSV gerado: {csv_path}")
    return csv_path

def main():
    # Verifica se existem ROIs salvas
    if not os.path.exists('last_rois.json'):
        print("Erro: Arquivo last_rois.json não encontrado!")
        print("Por favor, defina as ROIs primeiro usando a interface principal.")
        return
        
    # Carrega as ROIs salvas
    try:
        with open('last_rois.json', 'r') as f:
            rois = json.load(f)
    except Exception as e:
        print(f"Erro ao carregar ROIs: {e}")
        return
        
    if not rois:
        print("Erro: Nenhuma ROI definida no arquivo last_rois.json")
        print("Por favor, defina as ROIs primeiro usando a interface principal.")
        return
    
    # Diretório contendo os vídeos recortados
    videos_dir = "videos_recortados"
    if not os.path.exists(videos_dir):
        print(f"Diretório {videos_dir} não encontrado. Criando...")
        os.makedirs(videos_dir)
        print(f"Por favor, coloque os vídeos recortados no diretório {videos_dir}")
        return
    
    # Lista todos os vídeos no diretório
    video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not video_files:
        print(f"Nenhum vídeo encontrado em {videos_dir}")
        print("Formatos suportados: .mp4, .avi, .mov, .mkv")
        return
    
    print("\nROIs carregadas do arquivo last_rois.json:")
    for roi_type, roi_info in rois.items():
        print(f"{roi_type}: x={roi_info['x']:.3f}, y={roi_info['y']:.3f}, w={roi_info['width']:.3f}, h={roi_info['height']:.3f}")
    
    # Processa cada vídeo
    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        test_roi_extraction(video_path, rois)

if __name__ == "__main__":
    main() 