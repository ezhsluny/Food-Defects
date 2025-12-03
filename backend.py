from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os
import logging
import traceback
import time

# Импорт ваших моделей
try:
    from detect_deffects import AppleDefectDetector  
    from classification_defects import DefectClassifier
    from video_capture import advanced_preprocessing
    from make_bbox_apples import DetectorApples
    MODELS_LOADED = True
except ImportError as e:
    print(f"Модели не загружены: {e}")
    MODELS_LOADED = False

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Инициализация моделей
if MODELS_LOADED:
    try:
        detector_apples = DetectorApples()
        detector_defect = AppleDefectDetector("apples_defects_yolov8.pt")
        classifier = DefectClassifier('classificate_defect.onnx')
        print("Все модели успешно загружены")
    except Exception as e:
        print(f"Ошибка загрузки моделей: {e}")
        MODELS_LOADED = False

def ensure_bgr(image):
    """Гарантирует, что изображение в формате BGR для OpenCV"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    elif isinstance(image, Image.Image):
        image_np = np.array(image)
        if len(image_np.shape) == 3:
            return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return image_np
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

def ensure_rgb(image):
    """Гарантирует, что изображение в формате RGB для PIL и обработки"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    elif isinstance(image, Image.Image):
        return np.array(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

def image_to_base64(image):
    """Конвертирует изображение в base64 строку"""
    try:
        if isinstance(image, np.ndarray):
            # Конвертируем BGR в RGB для корректного отображения
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_rgb)
            else:
                image = Image.fromarray(image)
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
        return None

def process_image_with_models(image_np):
    """Обработка изображения с использованием моделей"""
    try:
        if not MODELS_LOADED:
            return {
                'apples_count': 2,
                'defects_count': 1,
                'detection_results': 'Модели не загружены - демо режим',
                'classification_results': {
                    'class': 'demo',
                    'confidence': 0.0
                },
                'processed_image': image_to_base64(image_np)
            }

        # Конвертируем в PIL для препроцессинга
        image_pil = Image.fromarray(ensure_rgb(image_np))
        
        # Препроцессинг
        processed_pil = advanced_preprocessing(image_pil)
        processed_np = np.array(processed_pil)
        processed_bgr = ensure_bgr(processed_np)
        
        # Детекция яблок
        crop_images, bboxes = detector_apples.crop_bboxes(processed_bgr)
        
        apples_count = len(crop_images)
        defects_count = 0
        classification_results = []
        
        # Обработка каждого обнаруженного яблока
        for i, crop in enumerate(crop_images):
            try:
                # Изменяем размер для моделей
                crop_resized = cv2.resize(crop, (100, 100))
                
                # Детекция дефектов
                defect_result = detector_defect.predict(crop_resized, save=False, return_image=True)
                
                # Классификация
                classification_result = classifier.predict(crop_resized)
                classification_results.append(classification_result)
                
                # Проверяем наличие дефектов
                if defect_result and isinstance(defect_result, dict):
                    if defect_result.get('has_defects', False):
                        defects_count += 1
                    elif 'defects' in defect_result and len(defect_result['defects']) > 0:
                        defects_count += 1
                        
            except Exception as e:
                logging.error(f"Error processing apple {i}: {e}")
                continue
        
        # Создаем визуализацию
        result_image = processed_bgr.copy()
        
        # Рисуем bounding boxes
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_image, f"Apple", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return {
            'apples_count': apples_count,
            'defects_count': defects_count,
            'detection_results': f'Обнаружено {apples_count} яблок, {defects_count} с дефектами',
            'classification_results': {
                'class': classification_results[0]['class'] if classification_results else 'unknown',
                'confidence': float(classification_results[0]['confidence']) if classification_results else 0.0
            },
            'processed_image': image_to_base64(result_image)
        }
        
    except Exception as e:
        logging.error(f"Error in process_image_with_models: {e}")
        traceback.print_exc()
        return {
            'error': f'Ошибка обработки: {str(e)}',
            'apples_count': 0,
            'defects_count': 0,
            'detection_results': 'Ошибка при обработке',
            'classification_results': {
                'class': 'error',
                'confidence': 0.0
            },
            'processed_image': image_to_base64(image_np)
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Читаем и конвертируем изображение
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        logging.info(f"Processing image: {file.filename}, size: {image_np.shape}")
        
        # Обработка изображения
        result = process_image_with_models(image_np)
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def process_video_with_models(video_path):
    """Обработка видео с использованием моделей"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Не удалось открыть видео файл'}
        
        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Создаем временный файл для обработанного видео
        output_path = "temp_processed_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        apples_total = 0
        defects_total = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Обрабатываем каждый кадр
            try:
                # Конвертируем в PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Препроцессинг
                processed_pil = advanced_preprocessing(frame_pil)
                
                processed_np = np.array(processed_pil)
                processed_bgr = ensure_bgr(processed_np)
                
                # Детекция яблок
                crop_images, bboxes = detector_apples.crop_bboxes(processed_bgr)
                
                frame_apples = len(crop_images)
                frame_defects = 0
                
                # Обработка каждого яблока
                for crop in crop_images:
                    try:
                        crop_resized = cv2.resize(crop, (100, 100))
                        defect_result = detector_defect.predict(crop_resized, save=False, return_image=True)
                        
                        if defect_result and isinstance(defect_result, dict):
                            if defect_result.get('has_defects', False) or ('defects' in defect_result and len(defect_result['defects']) > 0):
                                frame_defects += 1
                    except Exception as e:
                        continue
                
                apples_total += frame_apples
                defects_total += frame_defects
                
                # Рисуем bounding boxes и информацию на кадре
                result_frame = processed_bgr.copy()
                
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(result_frame, "Apple", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Добавляем статистику на кадр
                stats_text = [
                    f"Apples: {frame_apples}",
                    f"Defects: {frame_defects}",
                    f"Frame: {frame_count}/{total_frames}",
                    f"FPS: {fps}"
                ]
                
                for i, text in enumerate(stats_text):
                    cv2.putText(result_frame, text, (10, 30 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                out.write(result_frame)
                frame_count += 1
                
            except Exception as e:
                logging.error(f"Error processing frame {frame_count}: {e}")
                continue
        
        cap.release()
        out.release()
        
        processing_time = time.time() - start_time
        
        # Читаем обработанное видео и конвертируем в base64
        with open(output_path, 'rb') as f:
            video_data = f.read()
        video_base64 = base64.b64encode(video_data).decode()
        
        # Удаляем временный файл
        if os.path.exists(output_path):
            os.remove(output_path)
        
        return {
            'apples_count': apples_total,
            'defects_count': defects_total,
            'frames_processed': frame_count,
            'processing_time': processing_time,
            'detection_results': f'Обнаружено {apples_total} яблок, {defects_total} с дефектами',
            'processed_video': video_base64
        }
        
    except Exception as e:
        logging.error(f"Error in process_video_with_models: {e}")
        traceback.print_exc()
        return {'error': f'Ошибка обработки видео: {str(e)}'}

@app.route('/process-video', methods=['POST'])
def process_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Сохраняем временный файл
        temp_path = f"temp_{file.filename}"
        with open(temp_path, 'wb') as f:
            f.write(file.read())
        
        logging.info(f"Processing video: {file.filename}")
        
        # Обработка видео
        result = process_video_with_models(temp_path)
        
        # Удаляем временный файл
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        # Убедимся, что временный файл удален даже при ошибке
        temp_path = f"temp_{file.filename}"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Создаем папки если они не существуют
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print(f"Модели загружены: {MODELS_LOADED}")
    app.run(debug=True, host='0.0.0.0', port=5000)

@app.route('/process-realtime', methods=['POST'])
def process_realtime():
    """Обработка кадров в реальном времени"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Читаем и конвертируем изображение
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # # Конвертируем в RGB если нужно
        # image_np = convert_to_rgb(image_np)
        
        # Быстрая обработка для реального времени
        result = process_image_fast(image_np)
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error processing realtime frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_image_fast(image_np):
    """Быстрая обработка изображения для реального времени"""
    try:
        if not MODELS_LOADED:
            return {
                'apples_count': 2,
                'defects_count': 1,
                'classification_results': {
                    'class': 'demo',
                    'confidence': 0.0
                },
                'processed_image': image_to_base64(image_np)
            }

        # Быстрый препроцессинг
        image_pil = Image.fromarray(ensure_rgb(image_np))
        processed_pil = advanced_preprocessing(image_pil)
        processed_np = np.array(processed_pil)
        processed_bgr = ensure_bgr(processed_np)
        
        # Детекция яблок
        crop_images, bboxes = detector_apples.crop_bboxes(processed_bgr)
        
        apples_count = len(crop_images)
        defects_count = 0
        
        # Быстрая проверка дефектов (только для первого яблока для скорости)
        if crop_images:
            try:
                crop_resized = cv2.resize(crop_images[0], (100, 100))
                defect_result = detector_defect.predict(crop_resized, save=False, return_image=False)
                
                if defect_result and isinstance(defect_result, dict):
                    if defect_result.get('has_defects', False):
                        defects_count = 1
                    elif 'defects' in defect_result and len(defect_result['defects']) > 0:
                        defects_count = 1
            except Exception as e:
                logging.debug(f"Fast defect check failed: {e}")
        
        # Создаем визуализацию
        result_image = processed_bgr.copy()
        
        # Рисуем bounding boxes
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_image, "Apple", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return {
            'apples_count': apples_count,
            'defects_count': defects_count,
            'classification_results': {
                'class': 'realtime',
                'confidence': 0.9
            },
            'processed_image': image_to_base64(result_image)
        }
        
    except Exception as e:
        logging.error(f"Error in process_image_fast: {e}")
        return {
            'apples_count': 0,
            'defects_count': 0,
            'classification_results': {
                'class': 'error',
                'confidence': 0.0
            },
            'processed_image': image_to_base64(image_np)
        }