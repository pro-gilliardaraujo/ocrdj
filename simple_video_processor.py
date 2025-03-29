import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import pytesseract
import numpy as np
import pandas as pd
import json
import os
from PIL import Image, ImageTk
from pathlib import Path

# Configuração do Tesseract
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR'
if TESSERACT_PATH not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + TESSERACT_PATH
pytesseract.pytesseract.tesseract_cmd = os.path.join(TESSERACT_PATH, 'tesseract.exe')

class VideoProcessor:
    def __init__(self, master):
        self.master = master
        self.master.title('Processador de Vídeos')
        
        # Variáveis
        self.video_paths = []
        self.current_frame = None
        self.current_video = None
        self.rois = {}
        self.drawing = False
        self.start_point = None
        self.current_roi_type = None
        self.processing = False
        self.zoom_factor = 1.0
        self.frame_offset = (0, 0)  # Offset para centralizar a imagem
        
        # Define tamanho inicial da janela
        self.master.geometry('1280x720')
        
        # ROI Types e cores
        self.roi_types = {
            'Taxa de Aplicação': ('spray_rate', 'red'),
            'Altura': ('height', 'green'),
            'Espaçamento': ('spacing', 'blue'),
            'Velocidade': ('speed', 'yellow')
        }
        
        self.setup_ui()
        self.load_saved_rois()
    
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame esquerdo (lista de vídeos e controles)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Lista de vídeos
        ttk.Label(left_frame, text="Vídeos:").pack(anchor=tk.W)
        self.video_list = tk.Listbox(left_frame, width=40, height=10)
        self.video_list.pack(fill=tk.Y, expand=True)
        self.video_list.bind('<<ListboxSelect>>', self.on_video_selected)
        
        # Botões de controle de vídeo
        ttk.Button(left_frame, text="Adicionar Vídeos", command=self.add_videos).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="Remover Selecionado", command=self.remove_video).pack(fill=tk.X, pady=2)
        
        # Frame central
        center_frame = ttk.Frame(main_frame)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Frame para controles de zoom
        zoom_frame = ttk.Frame(center_frame)
        zoom_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(zoom_frame, text="Zoom +", command=lambda: self.adjust_zoom(1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Zoom -", command=lambda: self.adjust_zoom(0.8)).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Reset Zoom", command=lambda: self.adjust_zoom(1.0, True)).pack(side=tk.LEFT, padx=2)
        
        # Frame para o canvas com scrollbars
        canvas_frame = ttk.Frame(center_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(canvas_frame)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Canvas para o vídeo
        self.video_canvas = tk.Canvas(canvas_frame, bg='black',
                                    xscrollcommand=h_scrollbar.set,
                                    yscrollcommand=v_scrollbar.set)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Configura scrollbars
        h_scrollbar.config(command=self.video_canvas.xview)
        v_scrollbar.config(command=self.video_canvas.yview)
        
        # Binds do canvas
        self.video_canvas.bind('<Button-1>', self.start_roi)
        self.video_canvas.bind('<B1-Motion>', self.draw_roi)
        self.video_canvas.bind('<ButtonRelease-1>', self.end_roi)
        self.video_canvas.bind('<MouseWheel>', self.on_mousewheel)  # Windows
        self.video_canvas.bind('<Button-4>', self.on_mousewheel)    # Linux scroll up
        self.video_canvas.bind('<Button-5>', self.on_mousewheel)    # Linux scroll down
        
        # Frame para timeline
        timeline_frame = ttk.Frame(center_frame)
        timeline_frame.pack(fill=tk.X, pady=5)
        
        # Controles de navegação
        ttk.Button(timeline_frame, text="◀", command=lambda: self.navigate_frames(-1)).pack(side=tk.LEFT)
        self.frame_slider = ttk.Scale(timeline_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.frame_slider.bind('<ButtonRelease-1>', self.on_slider_changed)
        ttk.Button(timeline_frame, text="▶", command=lambda: self.navigate_frames(1)).pack(side=tk.LEFT)
        
        # Label para mostrar frame atual/total
        self.frame_label = ttk.Label(timeline_frame, text="Frame: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        
        # Frame direito (seleção de ROI e preview)
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        # Seleção de ROI
        ttk.Label(right_frame, text="Selecionar ROI:").pack(anchor=tk.W)
        self.roi_var = tk.StringVar()
        for roi_name in self.roi_types.keys():
            ttk.Radiobutton(right_frame, text=roi_name, 
                          variable=self.roi_var, value=roi_name,
                          command=self.on_roi_selected).pack(anchor=tk.W)
        
        # Preview da ROI
        ttk.Label(right_frame, text="Preview:").pack(anchor=tk.W, pady=(10,0))
        self.preview_canvas = tk.Canvas(right_frame, width=200, height=100, bg='black')
        self.preview_canvas.pack(pady=5)
        
        # Valor extraído
        self.value_label = ttk.Label(right_frame, text="Valor: --")
        self.value_label.pack(pady=5)
        
        # Botões de processamento
        ttk.Button(right_frame, text="Processar Vídeos", command=self.process_videos).pack(fill=tk.X, pady=2)
        self.stop_button = ttk.Button(right_frame, text="Parar", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        
        # Barra de progresso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.master, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Status
        self.status_label = ttk.Label(self.master, text="Pronto")
        self.status_label.pack(anchor=tk.W, padx=5)
    
    def add_videos(self):
        files = filedialog.askopenfilenames(
            title="Selecionar Vídeos",
            filetypes=[("Vídeos", "*.mp4 *.avi *.mkv"), ("Todos os arquivos", "*.*")]
        )
        if files:
            self.video_paths.extend(files)
            self.video_list.delete(0, tk.END)
            for path in self.video_paths:
                self.video_list.insert(tk.END, Path(path).name)
            
            if len(self.video_paths) == 1:
                self.load_video(0)
    
    def remove_video(self):
        selection = self.video_list.curselection()
        if selection:
            idx = selection[0]
            self.video_paths.pop(idx)
            self.video_list.delete(idx)
    
    def on_video_selected(self, event):
        selection = self.video_list.curselection()
        if selection:
            self.load_video(selection[0])
    
    def load_video(self, index):
        if 0 <= index < len(self.video_paths):
            if self.current_video is not None:
                self.current_video.release()
            
            self.current_video = cv2.VideoCapture(self.video_paths[index])
            if not self.current_video.isOpened():
                messagebox.showerror("Erro", "Não foi possível abrir o vídeo")
                return
            
            # Configura o tamanho inicial do canvas baseado no vídeo
            width = int(self.current_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.current_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Ajusta o tamanho do canvas para manter a proporção do vídeo
            canvas_width = min(width, 800)  # Limita a largura máxima
            canvas_height = int(height * (canvas_width / width))
            
            self.video_canvas.config(width=canvas_width, height=canvas_height)
            
            # Lê o primeiro frame
            ret, frame = self.current_video.read()
            if ret:
                self.current_frame = frame
                self.show_frame()
                
                # Atualiza o slider e label com o total de frames
                total_frames = int(self.current_video.get(cv2.CAP_PROP_FRAME_COUNT))
                self.frame_label.config(text=f"Frame: 1/{total_frames}")
                self.frame_slider.set(0)
            else:
                messagebox.showerror("Erro", "Não foi possível ler o frame do vídeo")
    
    def show_frame(self):
        if self.current_frame is None:
            return
        
        frame = self.current_frame.copy()
        h, w = frame.shape[:2]
        
        # Desenha ROIs existentes
        for roi_name, (roi_id, color) in self.roi_types.items():
            if roi_id in self.rois:
                roi = self.rois[roi_id]
                x1 = int(w * roi['x'])
                y1 = int(h * roi['y'])
                x2 = int(w * (roi['x'] + roi['width']))
                y2 = int(h * (roi['y'] + roi['height']))
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.get_color(color), 2)
        
        # Aplica zoom
        new_w = int(w * self.zoom_factor)
        new_h = int(h * self.zoom_factor)
        frame = cv2.resize(frame, (new_w, new_h))
        
        # Converte para formato Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        self.photo = ImageTk.PhotoImage(image=img)
        
        # Atualiza canvas e configura scrollregion
        self.video_canvas.delete("all")
        
        # Centraliza a imagem no canvas
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        x_offset = max(0, (canvas_width - new_w) // 2)
        y_offset = max(0, (canvas_height - new_h) // 2)
        
        self.video_canvas.create_image(x_offset, y_offset, image=self.photo, anchor=tk.NW)
        self.video_canvas.config(scrollregion=(0, 0, max(new_w, canvas_width), max(new_h, canvas_height)))
    
    def get_color(self, color_name):
        colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255)
        }
        return colors.get(color_name, (255, 255, 255))
    
    def canvas_to_frame_coords(self, canvas_x, canvas_y):
        """Converte coordenadas do canvas para coordenadas do frame original"""
        # Obtém o scroll atual
        scroll_x = self.video_canvas.xview()[0]
        scroll_y = self.video_canvas.yview()[0]
        
        # Ajusta as coordenadas considerando o scroll
        x = canvas_x + (scroll_x * self.video_canvas.winfo_width())
        y = canvas_y + (scroll_y * self.video_canvas.winfo_height())
        
        # Converte para coordenadas do frame original
        frame_x = x / self.zoom_factor
        frame_y = y / self.zoom_factor
        
        return frame_x, frame_y
    
    def start_roi(self, event):
        if not self.current_roi_type or self.current_frame is None:
            return
        
        self.drawing = True
        # Converte coordenadas do canvas para coordenadas do frame
        frame_x, frame_y = self.canvas_to_frame_coords(event.x, event.y)
        self.start_point = (frame_x, frame_y)
    
    def draw_roi(self, event):
        if not self.drawing or not self.start_point:
            return
        
        # Converte coordenadas do canvas para coordenadas do frame
        frame_x, frame_y = self.canvas_to_frame_coords(event.x, event.y)
        
        # Converte coordenadas do frame para coordenadas do canvas com zoom
        canvas_start_x = self.start_point[0] * self.zoom_factor
        canvas_start_y = self.start_point[1] * self.zoom_factor
        canvas_end_x = frame_x * self.zoom_factor
        canvas_end_y = frame_y * self.zoom_factor
        
        # Atualiza o frame com o retângulo atual
        self.show_frame()
        self.video_canvas.create_rectangle(
            canvas_start_x, canvas_start_y,
            canvas_end_x, canvas_end_y,
            outline=self.roi_types[self.current_roi_type][1]
        )
        
        # Extrai e mostra o valor em tempo real
        h, w = self.current_frame.shape[:2]
        x1 = int(min(self.start_point[0], frame_x))
        y1 = int(min(self.start_point[1], frame_y))
        x2 = int(max(self.start_point[0], frame_x))
        y2 = int(max(self.start_point[1], frame_y))
        
        # Garante que as coordenadas estejam dentro dos limites
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        if x2 > x1 and y2 > y1:  # Verifica se a ROI tem tamanho válido
            roi_img = self.current_frame[y1:y2, x1:x2]
            if roi_img.size > 0:
                # Atualiza o preview
                preview_img = cv2.resize(roi_img, (200, 100))
                preview_img = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(preview_img)
                self.preview_photo = ImageTk.PhotoImage(image=img)
                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(0, 0, image=self.preview_photo, anchor=tk.NW)
                
                # Extrai e mostra o valor
                roi_id = self.roi_types[self.current_roi_type][0]
                value = self.extract_value(roi_img, roi_id)
                if value is not None:
                    self.value_label.config(text=f"Valor: {value:.2f}")
                else:
                    self.value_label.config(text="Valor: --")
    
    def end_roi(self, event):
        if not self.drawing or not self.start_point:
            return
        
        self.drawing = False
        
        # Converte coordenadas do canvas para coordenadas do frame
        frame_x, frame_y = self.canvas_to_frame_coords(event.x, event.y)
        
        # Normaliza as coordenadas
        h, w = self.current_frame.shape[:2]
        x1 = min(self.start_point[0], frame_x) / w
        y1 = min(self.start_point[1], frame_y) / h
        x2 = max(self.start_point[0], frame_x) / w
        y2 = max(self.start_point[1], frame_y) / h
        
        # Limita as coordenadas entre 0 e 1
        x1 = max(0, min(1, x1))
        y1 = max(0, min(1, y1))
        x2 = max(0, min(1, x2))
        y2 = max(0, min(1, y2))
        
        roi_id = self.roi_types[self.current_roi_type][0]
        self.rois[roi_id] = {
            'x': x1,
            'y': y1,
            'width': x2 - x1,
            'height': y2 - y1
        }
        
        self.save_rois()
        self.show_frame()
        self.update_preview()
    
    def update_preview(self):
        if self.current_frame is None or not self.current_roi_type:
            return
        
        roi_id = self.roi_types[self.current_roi_type][0]
        if roi_id not in self.rois:
            return
        
        roi = self.rois[roi_id]
        h, w = self.current_frame.shape[:2]
        x1 = int(w * roi['x'])
        y1 = int(h * roi['y'])
        x2 = int(w * (roi['x'] + roi['width']))
        y2 = int(h * (roi['y'] + roi['height']))
        
        roi_img = self.current_frame[y1:y2, x1:x2]
        if roi_img.size == 0:
            return
        
        # Redimensiona para o preview
        roi_img = cv2.resize(roi_img, (200, 100))
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(roi_img)
        self.preview_photo = ImageTk.PhotoImage(image=img)
        
        self.preview_canvas.create_image(0, 0, image=self.preview_photo, anchor=tk.NW)
        
        # Extrai e mostra o valor
        value = self.extract_value(roi_img, roi_id)
        if value is not None:
            self.value_label.config(text=f"Valor: {value:.2f}")
        else:
            self.value_label.config(text="Valor: --")
    
    def extract_value(self, roi_img, roi_type):
        try:
            # Pré-processamento
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            
            # Ajusta o fator de escala baseado no tipo de ROI
            scale = 8 if roi_type in ['spacing', 'height'] else 7 if roi_type == 'speed' else 6
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Aplica diferentes técnicas de pré-processamento
            processed_images = []
            
            # 1. Threshold adaptativo com kernel maior
            norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            adaptive = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
            processed_images.append(adaptive)
            
            # 2. CLAHE com parâmetros ajustados
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
            cl1 = clahe.apply(gray)
            _, thresh_eq = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(thresh_eq)
            
            # Configuração do OCR
            config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.'
            
            # Tenta cada imagem processada
            for processed_img in processed_images:
                text = pytesseract.image_to_string(processed_img, config=config).strip()
                text = ''.join(c for c in text if c.isdigit() or c == '.')
                
                if text:
                    try:
                        value = float(text)
                        
                        # Ajustes específicos por tipo de ROI
                        if roi_type == 'height' and value > 20:
                            value /= 10
                        elif roi_type == 'spacing' and value > 5:
                            value /= 10
                        elif value > 100:
                            value /= 100
                        elif value > 10:
                            value /= 10
                        
                        # Validações específicas por tipo
                        if roi_type == 'spray_rate' and 0.1 <= value <= 10:
                            return value
                        elif roi_type == 'height' and 0.1 <= value <= 20:
                            return value
                        elif roi_type == 'spacing' and 0.1 <= value <= 5:
                            return value
                        elif roi_type == 'speed' and 0.1 <= value <= 20:
                            return value
                    except ValueError:
                        continue
            
            return None
        except Exception as e:
            print(f"Erro ao processar ROI: {str(e)}")
            return None
    
    def save_rois(self):
        try:
            with open('last_rois.json', 'w') as f:
                json.dump(self.rois, f, indent=4)
        except Exception as e:
            print(f"Erro ao salvar ROIs: {e}")
    
    def load_saved_rois(self):
        try:
            if os.path.exists('last_rois.json'):
                with open('last_rois.json', 'r') as f:
                    self.rois = json.load(f)
        except Exception as e:
            print(f"Erro ao carregar ROIs: {e}")
    
    def process_videos(self):
        if not self.video_paths:
            messagebox.showwarning("Aviso", "Nenhum vídeo selecionado")
            return
        
        if not self.rois:
            messagebox.showwarning("Aviso", "ROIs não definidas")
            return
        
        self.processing = True
        self.stop_button.config(state=tk.NORMAL)
        self.process_thread()
    
    def process_thread(self):
        if not self.processing:
            return
        
        results = []
        total_videos = len(self.video_paths)
        
        try:
            for video_idx, video_path in enumerate(self.video_paths):
                if not self.processing:
                    break
                
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    continue
                
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_interval = fps  # 1 frame por segundo
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_name = Path(video_path).stem
                
                for frame_count in range(0, total_frames, frame_interval):
                    if not self.processing:
                        break
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Processa o frame
                    frame_results = self.process_frame(frame, video_name, frame_count, fps)
                    if frame_results:
                        results.append(frame_results)
                    
                    # Atualiza progresso
                    progress = ((video_idx * total_frames) + frame_count) / (total_videos * total_frames) * 100
                    self.progress_var.set(progress)
                    self.status_label.config(text=f"Processando {video_name} - Frame {frame_count}/{total_frames}")
                    self.master.update_idletasks()
                
                cap.release()
            
            # Salva resultados
            if results:
                self.save_results(results)
            
            self.status_label.config(text="Processamento concluído")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro durante o processamento: {str(e)}")
        finally:
            self.processing = False
            self.stop_button.config(state=tk.DISABLED)
            self.progress_var.set(0)
    
    def process_frame(self, frame, video_name, frame_num, fps):
        results = {
            'Video': video_name,
            'Frame': frame_num,
            'Tempo (s)': frame_num / fps
        }
        
        h, w = frame.shape[:2]
        for roi_name, (roi_id, _) in self.roi_types.items():
            if roi_id in self.rois:
                roi = self.rois[roi_id]
                x1 = int(w * roi['x'])
                y1 = int(h * roi['y'])
                x2 = int(w * (roi['x'] + roi['width']))
                y2 = int(h * (roi['y'] + roi['height']))
                
                roi_img = frame[y1:y2, x1:x2]
                value = self.extract_value(roi_img, roi_id)
                
                column_name = {
                    'spray_rate': 'Taxa de Aplicação (L/min)',
                    'height': 'Altura (m)',
                    'spacing': 'Espaçamento (m)',
                    'speed': 'Velocidade (m/s)'
                }[roi_id]
                
                results[column_name] = value
        
        return results
    
    def save_results(self, results):
        df = pd.DataFrame(results)
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="resultados.csv"
        )
        
        if filename:
            df.to_csv(filename, 
                     index=False,
                     encoding='utf-8-sig',
                     sep=';',
                     decimal=',',
                     float_format='%.2f')
            
            messagebox.showinfo("Sucesso", f"Resultados salvos em {filename}")
    
    def stop_processing(self):
        self.processing = False
        self.stop_button.config(state=tk.DISABLED)
    
    def adjust_zoom(self, factor, reset=False):
        if reset:
            self.zoom_factor = 1.0
        else:
            self.zoom_factor *= factor
            # Limita o zoom entre 0.1x e 10x
            self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))
        
        self.show_frame()
    
    def on_mousewheel(self, event):
        if event.state & 4:  # Ctrl pressionado
            if event.delta > 0 or event.num == 4:
                self.adjust_zoom(1.1)
            else:
                self.adjust_zoom(0.9)
    
    def navigate_frames(self, delta):
        if not self.current_video:
            return
        
        current_pos = int(self.current_video.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(self.current_video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        new_pos = max(0, min(total_frames - 1, current_pos + delta))
        self.current_video.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        
        ret, frame = self.current_video.read()
        if ret:
            self.current_frame = frame
            self.show_frame()
            self.frame_slider.set((new_pos / total_frames) * 100)
            self.frame_label.config(text=f"Frame: {new_pos}/{total_frames}")
    
    def on_slider_changed(self, event):
        if not self.current_video:
            return
        
        total_frames = int(self.current_video.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = int((self.frame_slider.get() / 100) * total_frames)
        
        self.current_video.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = self.current_video.read()
        if ret:
            self.current_frame = frame
            self.show_frame()
            self.frame_label.config(text=f"Frame: {target_frame}/{total_frames}")
    
    def on_roi_selected(self):
        self.current_roi_type = self.roi_var.get()

def main():
    root = tk.Tk()
    app = VideoProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main() 