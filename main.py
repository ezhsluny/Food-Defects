from detect_deffects import AppleDefectDetector  
from classification_defects import DefectClassifier
from video_capture import advanced_preprocessing
from make_bbox_apples import DetectorApples

import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageEnhance
import time
import numpy as np
import os
from datetime import datetime
import shutil
import csv



def write_data(file_path, data):

    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'defect_severity', 'confidence', 'defect_area_percent']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(data)
        print(f"Данные записаны в CSV: {data}")



def ensure_bgr(image):
    """Гарантирует, что изображение в формате BGR для OpenCV"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Более надежная проверка: если среднее значение по каналу 0 больше чем по каналу 2 - вероятно RGB
            if np.mean(image[:,:,0]) > np.mean(image[:,:,2]):
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    elif isinstance(image, Image.Image):
        # Конвертируем PIL Image в BGR numpy array
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
            # Проверяем порядок каналов
            if image[0, 0, 0] < image[0, 0, 2]:  # Если B < R, вероятно BGR
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    elif isinstance(image, Image.Image):
        return np.array(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

def save_image_corrected(image, results_dir, index, timestamp, prefix="crop"):
    """Правильное сохранение изображения с корректными цветами"""
    try:
        save_path = os.path.join(results_dir, f"{prefix}_{index}_{timestamp}.jpg")
        
        if isinstance(image, np.ndarray):
            # Убеждаемся, что изображение в BGR для корректного сохранения
            image_to_save = ensure_bgr(image)
            cv2.imwrite(save_path, image_to_save)
            print(f"Изображение сохранено: {save_path}")
            return save_path
        elif isinstance(image, Image.Image):
            # PIL сохраняет в RGB
            image.save(save_path)
            print(f"Изображение сохранено: {save_path}")
            return save_path
    except Exception as e:
        print(f"Ошибка сохранения {save_path}: {e}")
    return False

if __name__ == '__main__':
    results_dir = "detection_results"
    csv_path = "defects.csv"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Папка для результатов: {os.path.abspath(results_dir)}")
    
    cv2.namedWindow("result")

    cap = cv2.VideoCapture('apples.mp4')
    #cap = cv2.VideoCapture(0)
    detectorApples = DetectorApples()
    detectorDefect = AppleDefectDetector("apples_defects_yolov8.pt")
    classifier = DefectClassifier('classificate_defect.onnx')

    video = int(input("Введите 1 для видео или 0 для изображения: "))

    if video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 100)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)
        
        prev_time = time.time()
        frame_count = 0
        fps_res = 0
        processed_count = 0

        while True:
            flag, img = cap.read()
            
            if not flag:
                print("Не удалось получить кадр с камеры")
                break
                
            try:
                frame_count += 1
                current_time = time.time()
                if current_time - prev_time >= 1.0:
                    fps_res = frame_count / (current_time - prev_time)
                    frame_count = 0
                    prev_time = current_time
                
                img = cv2.flip(img, 1)
                
   
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                
                processed_pil = advanced_preprocessing(pil_img)
        
                processed_np = np.array(processed_pil)
                processed_bgr = cv2.cvtColor(processed_np, cv2.COLOR_RGB2BGR)

                crop_images, bboxes = detectorApples.crop_bboxes(processed_bgr)
                if not crop_images or len(crop_images) == 0:
                    cv2.imshow('result', processed_bgr)
                    continue

                crop_images = [cv2.resize(crop, (100, 100)) for crop in crop_images]

                print(f"Найдено яблок: {len(crop_images)}")

                for i, crop in enumerate(crop_images):
                    processed_count += 1
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    
                    print(f"Обработка яблока {i+1}")

                    result = detectorDefect.predict(crop, save=True, return_image=True)
                    print(f"Детекция: {detectorDefect.get_detection_summary(result)}")

                    image_path = save_image_corrected(result['visualization'], results_dir, processed_count, timestamp, "detection_viz")
                    #1.Confidence
                    max_confidence = 0.0
                    if result['detections']:
                        for det in result['detections']:
                            current_confidence = det['confidence']
                            if current_confidence >  max_confidence:
                                max_confidence = current_confidence
                    
                    # 2. Степень дефекта 
                    defect_severity = "no_def"
                    severity_order = {'crit_def': 3, 'mid_def': 2, 'small_def': 1, 'no_def': 0}
                    for detection in result['detections']:
                        current_severity = detection['class_name']
                        if severity_order[current_severity] > severity_order[defect_severity]:
                            defect_severity = current_severity

                    
                    # 3. Площадь дефекта в % от общей площади изображения
                    total_image_area = crop.shape[0] * crop.shape[1]
                    total_defect_area = 0
                    for detection in result['detections']:
                        if detection['class_name'] != 'no_def':  
                            bbox_area = detection['bbox']['area']
                            total_defect_area += bbox_area
                    
                    defect_area_percent = (total_defect_area / total_image_area) * 100 if total_image_area > 0 else 0

                    # Записываем данные в CSV
                    csv_data = {
                        'filename': os.path.basename(image_path) if image_path else f"crop_{processed_count}_{timestamp}.jpg",
                        'defect_severity': defect_severity,
                        'confidence': max_confidence,
                        'defect_area_percent': round(defect_area_percent, 2),
                        
                    }
                    
                    write_data(csv_path, csv_data)

                    
                    if fps_res > 0:
                        cv2.putText(processed_bgr, f"FPS: {fps_res:.1f}", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(processed_bgr, f"Apples: {len(crop_images)}", (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow('result', processed_bgr)
                    
            except Exception as e:
                print(f"Произошла ошибка: {e}")
                import traceback
                traceback.print_exc()
                break
    
            ch = cv2.waitKey(1)  
            if ch == 27:  # ESC
                break
            
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Всего обработано: {processed_count} яблок")
        
    else:
        # Режим обработки одного изображения
        image_path = 'val/crit_def/photo_14949.jpg'
        print(f"Обработка изображения: {image_path}")
        
        # Загружаем изображение и сразу конвертируем в BGR
        test_image_bgr = cv2.imread(image_path)
        if test_image_bgr is not None:
            save_image_corrected(test_image_bgr, results_dir, "test", "original", "test_original")
        
        result1 = classifier.predict(image_path)
        print(f"Классификация: {result1['class']} ({result1['confidence']:.4f})")
        
        result = detectorDefect.predict(image_path, save=True, return_image=True)
        print(f"Детекция: {detectorDefect.get_detection_summary(result)}")
        
        # Сохраняем и отображаем результат
        if isinstance(result, dict) and 'visualization' in result:
            # Сохраняем визуализацию
            save_image_corrected(result['visualization'], results_dir, "single", datetime.now().strftime("%Y%m%d_%H%M%S"), "detection")
            
            # Для отображения конвертируем BGR в RGB
            result_image_rgb = cv2.cvtColor(ensure_bgr(result['visualization']), cv2.COLOR_BGR2RGB)
            cv2.imshow('Result', result_image_rgb)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
