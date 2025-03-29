import sys
import cv2
import pytesseract
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QFileDialog, QListWidget, QProgressBar, 
    QMessageBox, QFrame, QSplitter, QStyle, QStatusBar, QMenu, QAction, QTableWidget, QHeaderView, QTableWidgetItem, QLineEdit, QDialog, QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QSlider, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QWaitCondition, QMutex
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen

# Configuração do Tesseract
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR'
if TESSERACT_PATH not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + TESSERACT_PATH
pytesseract.pytesseract.tesseract_cmd = os.path.join(TESSERACT_PATH, 'tesseract.exe')

class CorrectionRule:
    def __init__(self, roi_type, min_value, max_value, corrected_value):
        self.roi_type = roi_type
        self.min_value = min_value
        self.max_value = max_value
        self.corrected_value = corrected_value
    
    def applies_to(self, value):
        if value is None:
            return False
        return self.min_value <= value <= self.max_value

class CorrectionRulesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Regras de Correção")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Lista de regras
        self.rules_list = QTableWidget()
        self.rules_list.setColumnCount(5)
        self.rules_list.setHorizontalHeaderLabels([
            "Tipo", "Mín", "Máx", "Valor Correto", "Ações"
        ])
        self.rules_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.rules_list)
        
        # Formulário para adicionar regra
        form_group = QGroupBox("Adicionar Regra")
        form_layout = QFormLayout()
        
        self.roi_type_combo = QComboBox()
        self.roi_type_combo.addItems([
            "Taxa de Aplicação",
            "Altura",
            "Espaçamento",
            "Velocidade"
        ])
        
        self.min_value = QDoubleSpinBox()
        self.min_value.setRange(0, 100)
        self.min_value.setDecimals(2)
        
        self.max_value = QDoubleSpinBox()
        self.max_value.setRange(0, 100)
        self.max_value.setDecimals(2)
        
        self.corrected_value = QDoubleSpinBox()
        self.corrected_value.setRange(0, 100)
        self.corrected_value.setDecimals(2)
        
        form_layout.addRow("Tipo:", self.roi_type_combo)
        form_layout.addRow("Valor Mínimo:", self.min_value)
        form_layout.addRow("Valor Máximo:", self.max_value)
        form_layout.addRow("Valor Correto:", self.corrected_value)
        
        form_group.setLayout(form_layout)
        layout.addWidget(form_group)
        
        # Botões
        buttons_layout = QHBoxLayout()
        
        add_button = QPushButton("Adicionar Regra")
        add_button.clicked.connect(self.add_rule)
        
        close_button = QPushButton("Fechar")
        close_button.clicked.connect(self.accept)
        
        buttons_layout.addWidget(add_button)
        buttons_layout.addWidget(close_button)
        layout.addLayout(buttons_layout)
        
        # Mapeamento de nomes para tipos de ROI
        self.roi_type_map = {
            "Taxa de Aplicação": "spray_rate",
            "Altura": "height",
            "Espaçamento": "spacing",
            "Velocidade": "speed"
        }
        
        self.rules = []
        self.update_rules_list()
    
    def add_rule(self):
        roi_name = self.roi_type_combo.currentText()
        roi_type = self.roi_type_map[roi_name]
        min_val = self.min_value.value()
        max_val = self.max_value.value()
        corrected = self.corrected_value.value()
        
        if min_val >= max_val:
            QMessageBox.warning(self, "Erro", "Valor mínimo deve ser menor que o máximo")
            return
        
        rule = CorrectionRule(roi_type, min_val, max_val, corrected)
        self.rules.append(rule)
        self.update_rules_list()
    
    def update_rules_list(self):
        self.rules_list.setRowCount(len(self.rules))
        for i, rule in enumerate(self.rules):
            # Encontra o nome amigável do tipo de ROI
            roi_name = next(key for key, value in self.roi_type_map.items() 
                          if value == rule.roi_type)
            
            self.rules_list.setItem(i, 0, QTableWidgetItem(roi_name))
            self.rules_list.setItem(i, 1, QTableWidgetItem(f"{rule.min_value:.2f}"))
            self.rules_list.setItem(i, 2, QTableWidgetItem(f"{rule.max_value:.2f}"))
            self.rules_list.setItem(i, 3, QTableWidgetItem(f"{rule.corrected_value:.2f}"))
            
            delete_button = QPushButton("Remover")
            delete_button.clicked.connect(lambda checked, row=i: self.delete_rule(row))
            self.rules_list.setCellWidget(i, 4, delete_button)
    
    def delete_rule(self, row):
        self.rules.pop(row)
        self.update_rules_list()

class VideoProcessor(QThread):
    progress_updated = pyqtSignal(int, str)
    frame_processed = pyqtSignal(np.ndarray, dict, int)
    processing_finished = pyqtSignal(pd.DataFrame)
    
    def __init__(self, video_paths, rois, parent=None):
        super().__init__(parent)
        self.video_paths = video_paths
        self.rois = rois
        self.running = True
        self.paused = False
        self.pause_condition = QWaitCondition()
        self.mutex = QMutex()
        self.frames_per_second = 1  # Reduzido para 1 frame por segundo
        self.correction_rules = []  # Lista de regras de correção
        
    def run(self):
        all_results = []
        total_videos = len(self.video_paths)
        
        for video_idx, video_path in enumerate(self.video_paths):
            if not self.running:
                break
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            
            # Obtém FPS do vídeo e calcula o intervalo de frames
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_interval = fps  # 1 frame por segundo
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Cria diretório para os resultados deste vídeo
            video_dir = os.path.join("resultados_temp", video_name)
            os.makedirs(video_dir, exist_ok=True)
            
            # Processa cada frame no intervalo definido
            for frame_count in range(0, total_frames, frame_interval):
                # Verifica se está pausado
                self.mutex.lock()
                if self.paused:
                    self.pause_condition.wait(self.mutex)
                self.mutex.unlock()
                
                if not self.running:
                    break
                
                # Define a posição do frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Cria diretório para o frame atual
                frame_dir = os.path.join(video_dir, f"frame_{frame_count}")
                os.makedirs(frame_dir, exist_ok=True)
                
                # Processa o frame
                results, debug_info = self.process_frame(frame, frame_count, frame_dir)
                self.frame_processed.emit(frame, results, frame_count)
                
                if results:
                    frame_data = {
                        'Video': video_name,
                        'Frame': frame_count,
                        'Tempo (s)': frame_count / fps,
                        'Taxa de Aplicação (L/min)': results.get('spray_rate', {}).get('value', None),
                        'Altura (m)': results.get('height', {}).get('value', None),
                        'Espaçamento (m)': results.get('spacing', {}).get('value', None),
                        'Velocidade (m/s)': results.get('speed', {}).get('value', None)
                    }
                    all_results.append(frame_data)
                
                # Calcula o progresso
                progress = (frame_count * 100) // total_frames
                video_progress = ((video_idx * 100) + progress) // total_videos
                
                self.progress_updated.emit(
                    video_progress,
                    f"Processando vídeo {video_idx + 1}/{total_videos} - Frame {frame_count}/{total_frames} - {frame_count//fps}s"
                )
            
            cap.release()
        
        if all_results:
            df = pd.DataFrame(all_results)
            # Remove outliers e valores impossíveis
            df = self.clean_results(df)
            self.processing_finished.emit(df)
    
    def process_frame(self, frame, frame_num, debug_dir):
        """
        Processa um frame do vídeo usando as ROIs definidas pelo usuário
        """
        if not self.rois:
            return {}, {}

        results = {}
        debug_info = {}
        
        h, w = frame.shape[:2]
        
        # Processa cada ROI definida pelo usuário
        for roi_type, roi_info in self.rois.items():
            if not roi_info:  # Pula se a ROI não foi definida
                continue
                
            # Calcula as coordenadas da ROI usando as definidas pelo usuário
            x1 = int(w * roi_info['x'])
            y1 = int(h * roi_info['y'])
            x2 = int(w * (roi_info['x'] + roi_info['width']))
            y2 = int(h * (roi_info['y'] + roi_info['height']))
            
            # Extrai a região da ROI
            roi_img = frame[y1:y2, x1:x2]
            
            # Cria diretório para as imagens da ROI
            roi_dir = os.path.join(debug_dir, roi_type)
            os.makedirs(roi_dir, exist_ok=True)
            
            # Salva a imagem original da ROI
            cv2.imwrite(os.path.join(roi_dir, "original.png"), roi_img)
            
            # Processa com diferentes técnicas
            preprocessed_images = self.enhance_image_for_ocr(roi_img, roi_type)
            if not preprocessed_images:
                continue
            
            debug_info[roi_type] = {}
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
                                if '.' not in text:
                                    if roi_type == 'height':
                                        # Para altura, se o valor for maior que 20, divide por 10
                                        if value > 20:
                                            value = value / 10
                                    elif value >= 100:
                                        value = value / 100
                                    elif value >= 10:
                                        value = value / 10
                                
                                # Validações específicas por tipo
                                is_valid = False
                                if roi_type == 'spray_rate' and 0.1 <= value <= 10:
                                    is_valid = True
                                elif roi_type == 'height' and 0.1 <= value <= 20:
                                    is_valid = True
                                elif roi_type == 'speed' and 0.1 <= value <= 20:
                                    is_valid = True
                            
                            confidence = 'high' if is_valid else 'low'
                            
                            # Salva os resultados desta técnica
                            debug_info[roi_type][technique] = {
                                'value': value,
                                'confidence': confidence,
                                'image': processed_img
                            }
                            
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
        
        return results, debug_info
    
    def enhance_image_for_ocr(self, image, roi_type):
        """
        Função otimizada para pré-processamento de imagem usando vários métodos
        """
        if image is None or image.size == 0:
            return None

        # Converte para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aumenta significativamente o fator de escala
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
    
    def clean_results(self, df):
        """Remove outliers e valores impossíveis do DataFrame"""
        # Define limites para cada coluna
        limits = {
            'Taxa de Aplicação (L/min)': (0.1, 10),
            'Altura (m)': (0.1, 5),
            'Espaçamento (m)': (0.1, 5),
            'Velocidade (m/s)': (0.1, 10)
        }
        
        # Remove valores fora dos limites
        for col, (min_val, max_val) in limits.items():
            if col in df.columns:
                df = df[df[col].between(min_val, max_val)]
        
        # Remove outliers usando IQR
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
        
        return df
    
    def pause(self):
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()
    
    def resume(self):
        self.mutex.lock()
        self.paused = False
        self.pause_condition.wakeAll()
        self.mutex.unlock()
    
    def stop(self):
        self.mutex.lock()
        self.running = False
        self.paused = False
        self.pause_condition.wakeAll()
        self.mutex.unlock()

class ROISelector(QWidget):
    roi_selected = pyqtSignal(str, dict)  # Sinal emitido quando uma ROI é selecionada
    
    def __init__(self):
        super().__init__()
        self.rois = {
            'spray_rate': {
                'name': 'Taxa de Aplicação',
                'color': QColor(255, 100, 100),  # Vermelho claro
                'roi': None
            },
            'height': {
                'name': 'Altura',
                'color': QColor(100, 255, 100),  # Verde claro
                'roi': None
            },
            'spacing': {
                'name': 'Espaçamento',
                'color': QColor(100, 100, 255),  # Azul claro
                'roi': None
            },
            'speed': {
                'name': 'Velocidade',
                'color': QColor(255, 255, 100),  # Amarelo claro
                'roi': None
            }
        }
        self.current_roi = None
        self.preview_label = None
        self.value_label = None
        self.main_window = None
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Grupo de botões para seleção de ROI
        roi_group = QGroupBox("Selecione a ROI")
        roi_layout = QVBoxLayout()
        
        # Adiciona label explicativo
        label = QLabel("Selecione a ROI para configurar:")
        roi_layout.addWidget(label)
        
        # Botões para ROIs individuais
        for roi_id in ['spray_rate', 'height', 'spacing', 'speed']:
            roi = self.rois[roi_id]
            btn = QPushButton(roi['name'])
            btn.setStyleSheet(f"""
                background-color: {roi['color'].name()};
                color: black;
                padding: 5px;
            """)
            btn.clicked.connect(lambda checked, x=roi_id: self.select_roi(x))
            roi_layout.addWidget(btn)

        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)
        
        # Preview da área selecionada
        preview_group = QFrame()
        preview_group.setStyleSheet("QFrame { background-color: #1a1a1a; border-radius: 5px; }")
        preview_layout = QVBoxLayout(preview_group)
        
        preview_title = QLabel("Preview da Área")
        preview_title.setStyleSheet("color: white; font-weight: bold;")
        preview_layout.addWidget(preview_title)
        
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(200, 100)
        self.preview_label.setStyleSheet("background-color: black;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.preview_label)
        
        # Valor extraído
        self.value_label = QLabel("Valor: --")
        self.value_label.setStyleSheet("color: white; font-size: 12px;")
        preview_layout.addWidget(self.value_label)
        
        layout.addWidget(preview_group)
        layout.addStretch()
        
        # Encontra a MainWindow pai
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, QMainWindow):
                self.main_window = parent
                break
            parent = parent.parent()
    
    def select_roi(self, roi_id):
        try:
            self.current_roi = roi_id
            if roi_id in self.rois:
                self.update_status()
                self.save_last_rois()
                self.roi_selected.emit(roi_id, self.rois[roi_id])
        except Exception as e:
            print(f"Erro ao selecionar ROI: {e}")
    
    def update_status(self):
        try:
            if self.current_roi and self.current_roi in self.rois:
                roi_info = self.rois[self.current_roi]
                self.preview_label.setText(f"Preview da ROI: {roi_info['name']}")
                if 'roi' in roi_info and roi_info['roi'] is not None:
                    self.value_label.setText(f"ROI definida: {roi_info['roi']}")
                else:
                    self.value_label.setText("ROI não definida")
            else:
                self.preview_label.setText("Nenhuma ROI selecionada")
                self.value_label.setText("Valor: --")
        except Exception as e:
            print(f"Erro ao atualizar status: {e}")
    
    def save_last_rois(self):
        try:
            rois_to_save = {}
            for roi_id, roi_info in self.rois.items():
                if 'roi' in roi_info and roi_info['roi'] is not None:
                    rois_to_save[roi_id] = roi_info['roi']
            
            if rois_to_save:
                with open('last_rois.json', 'w') as f:
                    json.dump(rois_to_save, f, indent=4)
                print("ROIs salvas com sucesso")
        except Exception as e:
            print(f"Erro ao salvar ROIs: {e}")
    
    def update_preview(self, frame, roi):
        try:
            if frame is None or roi is None:
                self.preview_label.clear()
                self.value_label.setText("Valor: --")
                return
            
            h, w = frame.shape[:2]
            x1 = int(w * roi['x'])
            y1 = int(h * roi['y'])
            x2 = int(w * (roi['x'] + roi['width']))
            y2 = int(h * (roi['y'] + roi['height']))
            
            roi_img = frame[y1:y2, x1:x2]
            if roi_img.size > 0:
                roi_img = cv2.resize(roi_img, (200, 100))
                roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
                h, w, ch = roi_img.shape
                bytes_per_line = ch * w
                image = QImage(roi_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.preview_label.setPixmap(QPixmap.fromImage(image))
                
                # Extrai e mostra o valor
                value = self.extract_value(roi_img)
                if value is not None:
                    self.value_label.setText(f"Valor: {value:.2f}")
                else:
                    self.value_label.setText("Valor: --")
        except Exception as e:
            print(f"Erro ao atualizar preview: {e}")
    
    def extract_value(self, image):
        try:
            if image is None or image.size == 0:
                return None
                
            # Pré-processamento para preview
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Aumenta significativamente o fator de escala
            scale = 7
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Aplica vários métodos de pré-processamento
            preprocessed_images = []
            
            # 1. Normalização básica com threshold adaptativo
            norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            adaptive1 = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
            preprocessed_images.append(adaptive1)
            
            # 2. Threshold Otsu com blur
            blur = cv2.GaussianBlur(gray, (3,3), 0)
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_images.append(otsu)
            
            # 3. Threshold adaptativo direto
            adaptive2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)
            preprocessed_images.append(adaptive2)
            
            # 4. Normalização com equalização de histograma
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = clahe.apply(gray)
            _, thresh_eq = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_images.append(thresh_eq)
            
            results = []
            for processed_img in preprocessed_images:
                custom_config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789. -c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -c tessedit_write_images=false'
                text = pytesseract.image_to_string(processed_img, config=custom_config).strip()
                text = ''.join(c for c in text if c.isdigit() or c == '.')
                
                if text and text.count('.') <= 1:
                    try:
                        value = float(text)
                        
                        # Se não tem ponto decimal e o número é grande, tenta interpretar como decimal
                        if '.' not in text:
                            if value >= 100:
                                value = value / 100
                            elif value >= 10:
                                value = value / 10
                        
                        results.append(value)
                    except ValueError:
                        continue
            
            # Se tiver resultados, retorna a mediana para maior precisão
            if results:
                # Remove outliers antes de calcular a mediana
                if len(results) > 2:
                    results.sort()
                    results = results[1:-1]  # Remove o menor e o maior valor
                return np.median(results)
            return None
        except Exception as e:
            print(f"Erro ao extrair valor: {e}")
            return None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Processador de Vídeos')
        self.setMinimumSize(1200, 800)
        
        # Inicializa variáveis
        self.video_paths = []
        self.cap = None
        self.current_frame = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.zoom_factor = 1.0
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.rois = {}
        self.frame_dims = (0, 0)
        self.view_dims = (0, 0)
        self.processor = None
        self.correction_rules = []
        self.paused = False
        self.is_roi_active = False  # Nova flag para controle de estado do ROI
        
        # Widgets que precisam ser acessados globalmente
        self.video_view = None
        self.scroll_area = None
        self.preview_label = None
        self.value_label = None
        self.progress_bar = None
        self.status_label = None
        self.data_table = None
        self.frame_slider = None
        self.frame_label = None
        self.btn_prev_frame = None
        self.btn_next_frame = None
        self.btn_start = None
        self.btn_pause = None
        self.btn_stop = None
        self.btn_adjust = None
        self.roi_selector = None
        
        # Inicializa a interface
        self.initUI()
        
        # Carrega ROIs salvas
        self.load_saved_rois()
        
        # Maximiza a janela
        self.showMaximized()
    
    def initUI(self):
        self.setWindowTitle('Processador de Vídeos')
        self.setMinimumSize(1200, 800)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # Painel esquerdo
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Lista de vídeos
        self.video_list = QListWidget()
        self.video_list.currentRowChanged.connect(self.on_video_selected)
        left_layout.addWidget(QLabel('Vídeos Selecionados:'))
        left_layout.addWidget(self.video_list)
        
        # Botões
        btn_add = QPushButton('Adicionar Vídeos')
        btn_add.clicked.connect(self.add_videos)
        btn_remove = QPushButton('Remover Selecionado')
        btn_remove.clicked.connect(self.remove_video)
        btn_clear = QPushButton('Limpar Lista')
        btn_clear.clicked.connect(self.clear_videos)
        
        left_layout.addWidget(btn_add)
        left_layout.addWidget(btn_remove)
        left_layout.addWidget(btn_clear)
        
        # Painel central
        central_panel = QWidget()
        central_layout = QVBoxLayout(central_panel)
        
        # Splitter vertical para dividir vídeo e tabela
        vertical_splitter = QSplitter(Qt.Vertical)
        
        # Área de visualização do vídeo com scroll
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        
        # Cria uma QScrollArea para o vídeo
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Widget para conter o vídeo
        self.video_container = QWidget()
        self.video_container_layout = QVBoxLayout(self.video_container)
        
        # Label do vídeo
        self.video_view = QLabel()
        self.video_view.setMinimumSize(640, 480)
        self.video_view.setAlignment(Qt.AlignCenter)
        self.video_view.setStyleSheet("background-color: black;")
        
        self.video_container_layout.addWidget(self.video_view)
        self.video_container_layout.addStretch()
        
        self.scroll_area.setWidget(self.video_container)
        video_layout.addWidget(self.scroll_area)
        
        # Controles de zoom
        zoom_controls = QHBoxLayout()
        
        btn_zoom_in = QPushButton("Zoom +")
        btn_zoom_in.clicked.connect(lambda: self.adjust_zoom(1.2))
        
        btn_zoom_out = QPushButton("Zoom -")
        btn_zoom_out.clicked.connect(lambda: self.adjust_zoom(0.8))
        
        btn_zoom_reset = QPushButton("Reset Zoom")
        btn_zoom_reset.clicked.connect(lambda: self.adjust_zoom(1.0, True))
        
        zoom_controls.addWidget(btn_zoom_in)
        zoom_controls.addWidget(btn_zoom_out)
        zoom_controls.addWidget(btn_zoom_reset)
        video_layout.addLayout(zoom_controls)
        
        # Barra de progresso e status
        self.progress_bar = QProgressBar()
        self.status_label = QLabel()
        video_layout.addWidget(self.progress_bar)
        video_layout.addWidget(self.status_label)
        
        vertical_splitter.addWidget(video_widget)
        
        # Tabela de dados
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        
        # Label para a tabela
        table_label = QLabel("Dados Extraídos em Tempo Real:")
        table_label.setStyleSheet("color: white; font-weight: bold;")
        table_layout.addWidget(table_label)
        
        # Cria a tabela
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(5)
        self.data_table.setHorizontalHeaderLabels([
            'Frame',
            'Taxa de Aplicação (L/min)',
            'Altura (m)',
            'Espaçamento (m)',
            'Velocidade (m/s)'
        ])
        # Define altura da linha para melhor visualização
        self.data_table.verticalHeader().setDefaultSectionSize(40)
        # Esconde o cabeçalho vertical
        self.data_table.verticalHeader().setVisible(False)
        # Ajusta as colunas
        self.data_table.horizontalHeader().setStretchLastSection(True)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # Estilo para melhor visualização
        self.data_table.setStyleSheet("""
            QTableWidget {
                background-color: #2d2d2d;
                gridline-color: #3d3d3d;
            }
            QTableWidget::item {
                padding: 5px;
                font-size: 14px;
            }
            QHeaderView::section {
                background-color: #3d3d3d;
                padding: 5px;
                font-size: 12px;
                font-weight: bold;
                border: 1px solid #4d4d4d;
            }
        """)
        table_layout.addWidget(self.data_table)
        
        vertical_splitter.addWidget(table_widget)
        
        # Define proporção inicial entre vídeo e tabela (70/30)
        vertical_splitter.setSizes([700, 300])
        
        central_layout.addWidget(vertical_splitter)
        
        # Adiciona controles de processamento
        self.processing_controls = QFrame()
        self.processing_controls.setStyleSheet("QFrame { background-color: #1a1a1a; border-radius: 5px; }")
        processing_layout = QVBoxLayout(self.processing_controls)
        
        # Botões com ícones
        self.btn_start = QPushButton('▶ Iniciar Processamento')
        self.btn_pause = QPushButton('⏸ Pausar')
        self.btn_stop = QPushButton('⏹ Parar')
        self.btn_adjust = QPushButton('⚙ Ajustar ROIs')
        
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_adjust.setEnabled(False)
        
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_pause.clicked.connect(self.pause_processing)
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_adjust.clicked.connect(self.adjust_rois)
        
        processing_layout.addWidget(self.btn_start)
        processing_layout.addWidget(self.btn_pause)
        processing_layout.addWidget(self.btn_stop)
        processing_layout.addWidget(self.btn_adjust)
        
        central_layout.addWidget(self.processing_controls)
        
        # Adiciona os painéis ao layout principal
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(central_panel)
        
        # Painel direito (Seletor de ROIs)
        self.roi_selector = ROISelector()
        self.roi_selector.roi_selected.connect(self.on_roi_selected)
        splitter.addWidget(self.roi_selector)
        splitter.setSizes([200, 800, 200])
        
        layout.addWidget(splitter)
        
        # Barra de status
        self.statusBar().showMessage('Pronto')
        
        # Menu
        self.create_menu()
        
        # Adiciona controles de navegação de frames
        navigation_widget = QWidget()
        navigation_layout = QHBoxLayout(navigation_widget)
        
        # Botão anterior
        self.btn_prev_frame = QPushButton("◀ Frame Anterior")
        self.btn_prev_frame.clicked.connect(lambda: self.navigate_frames(-1))
        self.btn_prev_frame.setEnabled(False)
        
        # Slider para timeline
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        
        # Botão próximo
        self.btn_next_frame = QPushButton("Frame Seguinte ▶")
        self.btn_next_frame.clicked.connect(lambda: self.navigate_frames(1))
        self.btn_next_frame.setEnabled(False)
        
        # Label para mostrar o frame atual
        self.frame_label = QLabel("Frame: 0/0")
        
        navigation_layout.addWidget(self.btn_prev_frame)
        navigation_layout.addWidget(self.frame_slider)
        navigation_layout.addWidget(self.btn_next_frame)
        navigation_layout.addWidget(self.frame_label)
        
        # Adiciona os controles de navegação ao layout central
        central_layout.addWidget(navigation_widget)
    
    def create_menu(self):
        menubar = self.menuBar()
        
        # Menu Arquivo
        file_menu = menubar.addMenu('Arquivo')
        
        load_roi_action = QAction('Carregar ROIs', self)
        load_roi_action.triggered.connect(self.load_rois)
        file_menu.addAction(load_roi_action)
        
        save_roi_action = QAction('Salvar ROIs', self)
        save_roi_action.triggered.connect(self.save_rois)
        file_menu.addAction(save_roi_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Sair', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
    
    def add_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Selecionar Vídeos",
            "",
            "Vídeos (*.mp4 *.avi *.mkv);;Todos os arquivos (*.*)"
        )
        
        if files:
            self.video_paths.extend(files)
            self.video_list.clear()
            self.video_list.addItems([Path(f).name for f in self.video_paths])
            
            if len(self.video_paths) == 1:
                self.load_video(0)
    
    def remove_video(self):
        current = self.video_list.currentRow()
        if current >= 0:
            self.video_paths.pop(current)
            self.video_list.takeItem(current)
    
    def clear_videos(self):
        self.video_paths.clear()
        self.video_list.clear()
    
    def on_video_selected(self, index):
        if index >= 0:
            self.load_video(index)
    
    def load_video(self, index):
        if 0 <= index < len(self.video_paths):
            # Fecha o vídeo anterior se estiver aberto
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.video_paths[index])
            if self.cap.isOpened():
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.current_frame_idx = 0
                
                # Atualiza o slider
                self.frame_slider.setMaximum(self.total_frames - 1)
                self.frame_slider.setValue(0)
                
                # Atualiza o label
                self.frame_label.setText(f"Frame: {self.current_frame_idx + 1}/{self.total_frames}")
                
                # Habilita os botões de navegação
                self.btn_prev_frame.setEnabled(True)
                self.btn_next_frame.setEnabled(True)
                
                # Carrega o primeiro frame
                self.load_frame(0)
    
    def load_frame(self, frame_idx):
        if self.cap is None or not self.cap.isOpened():
            return
        
        # Define a posição do frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame = frame.copy()
            self.current_frame_idx = frame_idx
            self.show_frame(frame)
            self.frame_label.setText(f"Frame: {frame_idx + 1}/{self.total_frames}")
            
            # Atualiza o slider se necessário
            if self.frame_slider.value() != frame_idx:
                self.frame_slider.setValue(frame_idx)
    
    def navigate_frames(self, delta):
        new_idx = self.current_frame_idx + delta
        if 0 <= new_idx < self.total_frames:
            self.load_frame(new_idx)
    
    def on_slider_changed(self):
        frame_idx = self.frame_slider.value()
        if frame_idx != self.current_frame_idx:
            self.load_frame(frame_idx)
    
    def keyPressEvent(self, event):
        # Adiciona navegação por teclado
        if event.key() == Qt.Key_Left:
            self.navigate_frames(-1)
        elif event.key() == Qt.Key_Right:
            self.navigate_frames(1)
        elif event.key() == Qt.Key_PageUp:
            self.navigate_frames(-10)
        elif event.key() == Qt.Key_PageDown:
            self.navigate_frames(10)
        else:
            super().keyPressEvent(event)
    
    def show_frame(self, frame, draw_selection=True):
        if frame is None:
            return
            
        self.current_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Calcula as novas dimensões com base no zoom
        new_w = int(w * self.zoom_factor)
        new_h = int(h * self.zoom_factor)
        
        # Desenha as ROIs existentes e a seleção atual
        frame_with_rois = frame.copy()
        
        # Desenha as ROIs existentes com cores diferentes e labels
        for roi_id, roi_info in self.rois.items():
            if roi_info:  # Verifica se a ROI existe
                color = self.roi_selector.rois[roi_id]['color'].getRgb()[:3]
                color = (color[2], color[1], color[0])  # Converte BGR para RGB
                
                # Calcula as coordenadas da ROI
                x1 = int(w * roi_info['x'])
                y1 = int(h * roi_info['y'])
                x2 = int(w * (roi_info['x'] + roi_info['width']))
                y2 = int(h * (roi_info['y'] + roi_info['height']))
                
                # Desenha o retângulo da ROI
                cv2.rectangle(frame_with_rois, (x1, y1), (x2, y2), color, 2)
                
                # Adiciona label com o nome da ROI
                label = self.roi_selector.rois[roi_id]['name']
                cv2.putText(frame_with_rois, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Desenha a seleção atual com linha fina
        if draw_selection and self.drawing and self.start_point and self.end_point:
            if self.roi_selector.current_roi:
                color = self.roi_selector.rois[self.roi_selector.current_roi]['color'].getRgb()[:3]
                color = (color[2], color[1], color[0])  # Converte BGR para RGB
                cv2.rectangle(frame_with_rois, 
                            self.start_point,
                            self.end_point,
                            color, 1)
        
        # Redimensiona o frame com o zoom atual
        frame_with_rois = cv2.resize(frame_with_rois, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        frame_with_rois = cv2.cvtColor(frame_with_rois, cv2.COLOR_BGR2RGB)
        
        image = QImage(frame_with_rois.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        self.video_view.setPixmap(QPixmap.fromImage(image))
        
        # Atualiza as dimensões do video_view
        self.video_view.setFixedSize(new_w, new_h)
        
        # Atualiza as dimensões para uso nos eventos do mouse
        self.frame_dims = (w, h)
        self.view_dims = (new_w, new_h)
    
    def mousePressEvent(self, event):
        try:
            if not self.roi_selector or not self.roi_selector.current_roi or not self.is_roi_active:
                return
                
            pos = self.video_view.mapFrom(self, event.pos())
            if not self.video_view.rect().contains(pos):
                return
                
            frame_pos = self.view_to_frame_coords(pos)
            if frame_pos:
                self.drawing = True
                self.start_point = frame_pos
                self.end_point = frame_pos
                
                if hasattr(self, 'current_frame') and self.current_frame is not None:
                    self.show_frame(self.current_frame.copy())
        except Exception as e:
            print(f"Erro no mousePressEvent: {e}")
            self.reset_roi_state()

    def mouseReleaseEvent(self, event):
        try:
            if not self.drawing:
                return
                
            self.drawing = False
            
            pos = self.video_view.mapFrom(self, event.pos())
            if not self.video_view.rect().contains(pos):
                self.reset_roi_state()
                return
                
            frame_pos = self.view_to_frame_coords(pos)
            if frame_pos and self.start_point and self.roi_selector and self.roi_selector.current_roi:
                self.end_point = frame_pos
                
                w, h = self.frame_dims
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                
                roi = {
                    'x': min(x1, x2) / w,
                    'y': min(y1, y2) / h,
                    'width': abs(x2 - x1) / w,
                    'height': abs(y2 - y1) / h
                }
                
                self.rois[self.roi_selector.current_roi] = roi
                self.roi_selector.rois[self.roi_selector.current_roi]['roi'] = roi
                self.roi_selector.update_status()
                self.roi_selector.save_last_rois()
                
                if self.current_frame is not None:
                    self.show_frame(self.current_frame.copy())
                    self.roi_selector.update_preview(self.current_frame, roi)
            
            self.reset_roi_state()
        except Exception as e:
            print(f"Erro no mouseReleaseEvent: {e}")
            self.reset_roi_state()

    def reset_roi_state(self):
        """Reseta o estado do ROI para evitar travamentos"""
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.is_roi_active = False
        if self.current_frame is not None:
            self.show_frame(self.current_frame.copy())

    def on_roi_selected(self, roi_id, roi):
        """Atualizado para melhor gerenciamento de estado"""
        self.is_roi_active = True
        self.rois[roi_id] = roi
        self.statusBar().showMessage(f'ROI {roi_id} selecionada. Clique e arraste para definir a área.')
        
        # Reset do estado ao selecionar nova ROI
        self.drawing = False
        self.start_point = None
        self.end_point = None

    def view_to_frame_coords(self, pos):
        """Converte coordenadas da view para coordenadas do frame considerando o zoom"""
        try:
            if not hasattr(self, 'frame_dims') or not hasattr(self, 'view_dims'):
                return None
            
            frame_w, frame_h = self.frame_dims
            view_w, view_h = self.view_dims
            
            if not hasattr(self, 'scroll_area'):
                return None
                
            # Obtém a posição do scroll
            scroll_x = self.scroll_area.horizontalScrollBar().value()
            scroll_y = self.scroll_area.verticalScrollBar().value()
            
            # Primeiro ajusta as coordenadas do mouse em relação ao centro da view
            x = pos.x() + scroll_x - (view_w - frame_w * self.zoom_factor) / 2
            y = pos.y() + scroll_y - (view_h - frame_h * self.zoom_factor) / 2
            
            # Depois converte para coordenadas do frame original usando o zoom_factor
            if hasattr(self, 'zoom_factor'):
                x = x / self.zoom_factor
                y = y / self.zoom_factor
                
                # Limita às dimensões do frame
                x = max(0, min(x, frame_w - 1))
                y = max(0, min(y, frame_h - 1))
                
                return (int(x), int(y))
            return None
        except Exception as e:
            print(f"Erro ao converter coordenadas: {e}")
            return None
    
    def wheelEvent(self, event):
        """Permite zoom com a roda do mouse enquanto segura Ctrl"""
        if event.modifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.adjust_zoom(1.1)
            else:
                self.adjust_zoom(0.9)
            event.accept()
        else:
            super().wheelEvent(event)
    
    def load_rois(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Carregar ROIs",
            "",
            "Arquivos JSON (*.json);;Todos os arquivos (*.*)"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    self.rois = json.load(f)
                self.statusBar().showMessage('ROIs carregadas com sucesso')
            except Exception as e:
                QMessageBox.warning(self, 'Erro', f'Erro ao carregar ROIs: {str(e)}')
    
    def save_rois(self):
        if not self.rois:
            QMessageBox.warning(self, 'Aviso', 'Nenhuma ROI para salvar')
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar ROIs",
            "",
            "Arquivos JSON (*.json);;Todos os arquivos (*.*)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.rois, f, indent=4)
                self.statusBar().showMessage('ROIs salvas com sucesso')
            except Exception as e:
                QMessageBox.warning(self, 'Erro', f'Erro ao salvar ROIs: {str(e)}')
    
    def show_correction_rules(self):
        dialog = CorrectionRulesDialog(self)
        dialog.rules = self.correction_rules.copy()
        dialog.update_rules_list()
        
        if dialog.exec_() == QDialog.Accepted:
            self.correction_rules = dialog.rules.copy()
            if self.processor:
                self.processor.correction_rules = self.correction_rules
    
    def start_processing(self):
        if not self.video_paths:
            QMessageBox.warning(self, 'Aviso', 'Nenhum vídeo selecionado')
            return
            
        if not self.rois:
            QMessageBox.warning(self, 'Aviso', 'ROIs não definidas')
            return
        
        # Limpa a tabela antes de começar novo processamento
        self.data_table.setRowCount(0)
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        self.processor = VideoProcessor(self.video_paths, self.rois)
        self.processor.correction_rules = self.correction_rules  # Passa as regras de correção
        self.processor.progress_updated.connect(self.update_progress)
        self.processor.frame_processed.connect(self.on_frame_processed)
        self.processor.processing_finished.connect(self.on_processing_finished)
        self.processor.start()
    
    def stop_processing(self):
        if self.processor:
            self.processor.stop()
            self.btn_stop.setEnabled(False)
    
    def update_progress(self, value, status):
        self.progress_bar.setValue(value)
        self.status_label.setText(status)
    
    def on_frame_processed(self, frame, results, frame_number):
        self.show_frame(frame)
        
        # Adiciona uma nova linha com os valores do frame atual
        if any(value is not None for value in results.values()):
            row_position = self.data_table.rowCount()
            self.data_table.insertRow(row_position)
            
            # Adiciona o número do frame
            frame_item = QTableWidgetItem(str(frame_number))
            frame_item.setTextAlignment(Qt.AlignCenter)
            frame_item.setBackground(QColor(70, 70, 70))
            self.data_table.setItem(row_position, 0, frame_item)
            
            # Mapeia os nomes das ROIs para as colunas da tabela
            column_mapping = {
                'spray_rate': 1,  # Taxa de Aplicação
                'height': 2,      # Altura
                'spacing': 3,     # Espaçamento
                'speed': 4        # Velocidade
            }
            
            # Preenche a linha com os valores do frame atual
            for roi_name, value_info in results.items():
                column = column_mapping.get(roi_name)
                if column is not None:
                    value = value_info.get('value')
                    confidence = value_info.get('confidence', 'low')
                    
                    item = QTableWidgetItem(f"{value:.2f}" if value is not None else "--")
                    item.setTextAlignment(Qt.AlignCenter)
                    
                    # Define cor de fundo baseada na confiança
                    if value is not None:
                        if confidence == 'high':
                            item.setBackground(QColor(200, 255, 200))  # Verde claro para alta confiança
                        else:
                            item.setBackground(QColor(255, 255, 200))  # Amarelo claro para baixa confiança
                    else:
                        item.setBackground(QColor(255, 200, 200))  # Vermelho claro para valores não reconhecidos
                    
                    self.data_table.setItem(row_position, column, item)
            
            # Rola a tabela para a última linha
            self.data_table.scrollToBottom()
        
        # Atualiza preview se houver ROI selecionada
        if self.roi_selector.current_roi:
            roi = self.roi_selector.rois[self.roi_selector.current_roi]['roi']
            self.roi_selector.update_preview(frame, roi)
    
    def on_processing_finished(self, df):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        if not df.empty:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Salvar Resultados",
                "resultados.csv",
                "Arquivos CSV (*.csv);;Todos os arquivos (*.*)"
            )
            
            if filename:
                # Salva o DataFrame com as configurações adequadas
                df.to_csv(filename, 
                         index=False,
                         encoding='utf-8-sig',  # Para suportar caracteres especiais
                         sep=';',               # Separador ponto e vírgula
                         decimal=',',           # Usa vírgula como separador decimal
                         float_format='%.2f')   # Formata números com 2 casas decimais
                
                # Salva também um arquivo de log com informações detalhadas
                log_filename = filename.replace('.csv', '_log.txt')
                with open(log_filename, 'w', encoding='utf-8') as f:
                    f.write("Log de Processamento\n")
                    f.write("===================\n\n")
                    f.write(f"Total de frames processados: {len(df)}\n")
                    f.write(f"Total de vídeos processados: {len(df['Video'].unique())}\n\n")
                    
                    # Estatísticas por tipo de ROI
                    for col in ['Taxa de Aplicação (L/min)', 'Altura (m)', 'Espaçamento (m)', 'Velocidade (m/s)']:
                        if col in df.columns:
                            values = df[col].dropna()
                            if not values.empty:
                                f.write(f"\nEstatísticas para {col}:\n")
                                f.write(f"  - Média: {values.mean():.2f}\n")
                                f.write(f"  - Mediana: {values.median():.2f}\n")
                                f.write(f"  - Desvio Padrão: {values.std():.2f}\n")
                                f.write(f"  - Mínimo: {values.min():.2f}\n")
                                f.write(f"  - Máximo: {values.max():.2f}\n")
                                f.write(f"  - Total de valores válidos: {len(values)}\n")
                                f.write(f"  - Taxa de sucesso: {(len(values) / len(df) * 100):.1f}%\n")
                
                self.statusBar().showMessage(f'Processamento concluído! Resultados salvos em {filename} e {log_filename}')
        else:
            self.statusBar().showMessage('Processamento concluído sem resultados.')
    
    def pause_processing(self):
        if self.processor:
            if not hasattr(self, 'paused') or not self.paused:
                self.processor.pause()
                self.btn_pause.setText('▶ Continuar')
                self.btn_adjust.setEnabled(True)
                self.paused = True
            else:
                self.processor.resume()
                self.btn_pause.setText('⏸ Pausar')
                self.btn_adjust.setEnabled(False)
                self.paused = False
    
    def adjust_rois(self):
        if self.paused:
            # Permite ajustar ROIs durante a pausa
            self.roi_selector.setEnabled(True)
            self.statusBar().showMessage('Ajuste as ROIs e clique em Continuar quando terminar')
    
    def load_saved_rois(self):
        """Carrega as ROIs salvas do arquivo last_rois.json"""
        try:
            if os.path.exists('last_rois.json'):
                with open('last_rois.json', 'r') as f:
                    saved_rois = json.load(f)
                if saved_rois:
                    self.rois = saved_rois
                    # Atualiza as ROIs no seletor
                    for roi_id, roi in saved_rois.items():
                        if roi_id in self.roi_selector.rois:
                            self.roi_selector.rois[roi_id]['roi'] = roi
                    self.roi_selector.update_status()
                    self.statusBar().showMessage('ROIs carregadas com sucesso')
        except Exception as e:
            print(f"Erro ao carregar ROIs: {e}")
            self.statusBar().showMessage('Erro ao carregar ROIs salvas')

    def adjust_zoom(self, factor, reset=False):
        """Ajusta o zoom da visualização do vídeo"""
        if reset:
            self.zoom_factor = 1.0
        else:
            self.zoom_factor *= factor
        
        # Limita o zoom entre 0.1x e 10x
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))
        
        if self.current_frame is not None:
            self.show_frame(self.current_frame)

    def mouseMoveEvent(self, event):
        try:
            if not self.drawing or not self.is_roi_active:
                return
                
            pos = self.video_view.mapFrom(self, event.pos())
            if not self.video_view.rect().contains(pos):
                return
                
            frame_pos = self.view_to_frame_coords(pos)
            if frame_pos and self.current_frame is not None:
                self.end_point = frame_pos
                self.show_frame(self.current_frame.copy())
                
                # Atualiza o preview em tempo real
                if self.roi_selector and self.roi_selector.current_roi and self.start_point:
                    w, h = self.frame_dims
                    x1, y1 = self.start_point
                    x2, y2 = self.end_point
                    temp_roi = {
                        'x': min(x1, x2) / w,
                        'y': min(y1, y2) / h,
                        'width': abs(x2 - x1) / w,
                        'height': abs(y2 - y1) / h
                    }
                    self.roi_selector.update_preview(self.current_frame, temp_roi)
        except Exception as e:
            print(f"Erro no mouseMoveEvent: {e}")
            self.reset_roi_state()

def main():
    # Verifica se o Tesseract está instalado
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        print(f"Erro: Tesseract-OCR não encontrado em {TESSERACT_PATH}")
        return

    app = QApplication(sys.argv)
    
    # Aplica estilo moderno
    app.setStyle('Fusion')
    
    # Tema escuro moderno
    palette = app.palette()
    palette.setColor(palette.Window, QColor(53, 53, 53))
    palette.setColor(palette.WindowText, Qt.white)
    palette.setColor(palette.Base, QColor(25, 25, 25))
    palette.setColor(palette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ToolTipBase, Qt.white)
    palette.setColor(palette.ToolTipText, Qt.white)
    palette.setColor(palette.Text, Qt.white)
    palette.setColor(palette.Button, QColor(53, 53, 53))
    palette.setColor(palette.ButtonText, Qt.white)
    palette.setColor(palette.BrightText, Qt.red)
    palette.setColor(palette.Link, QColor(42, 130, 218))
    palette.setColor(palette.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 