from detect_deffects import AppleDefectDetector  
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

def ensure_bgr(image):
    """Гарантирует, что изображение в формате BGR для OpenCV"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            if np.mean(image[:,:,0]) > np.mean(image[:,:,2]):
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
            if image[0, 0, 0] < image[0, 0, 2]: 
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    elif isinstance(image, Image.Image):
        return np.array(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

def save_image_corrected(image, results_dir, index, timestamp, prefix="crop"):
    """Сохранение изображения с корректными цветами"""
    try:
        save_path = os.path.join(results_dir, f"{prefix}_{index}_{timestamp}.jpg")
        
        if isinstance(image, np.ndarray):

            image_to_save = ensure_bgr(image)
            cv2.imwrite(save_path, image_to_save)
            print(f"Изображение сохранено: {save_path}")
            return True
        elif isinstance(image, Image.Image):
            # PIL сохраняет в RGB
            image.save(save_path)
            print(f"Изображение сохранено: {save_path}")
            return True
    except Exception as e:
        print(f"Ошибка сохранения {save_path}: {e}")
    return False


if __name__ == '__main__':
    results_dir = "detection_results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Папка для результатов: {os.path.abspath(results_dir)}")
    
    cv2.namedWindow("result")

    cap = cv2.VideoCapture('apples.mp4')

    detectorApples = DetectorApples()
    detectorDefect = AppleDefectDetector("apples_defects_yolov8.pt")


    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
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

                save_image_corrected(result['visualization'], results_dir, processed_count, timestamp, "detection_viz")
  
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
        
