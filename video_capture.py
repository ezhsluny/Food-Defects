import cv2
import time
import numpy as np
from PIL import Image, ImageEnhance
import os

# Кэш для хранения предыдущих результатов анализа
_prev_analysis = None
_prev_analysis_frame = None

def analyze_histogram_needs(img_array):
    global _prev_analysis, _prev_analysis_frame
    
    #Если кадр похож на предыдущий, используем кэш
    if _prev_analysis_frame is not None:
        frame_diff = np.mean(np.abs(img_array - _prev_analysis_frame))
        if frame_diff < 5:  
            return _prev_analysis
    
    mean_brightness = np.mean(img_array)
    std_brightness = np.std(img_array)
    
    
    sampled = img_array[::3, ::3]
    hist, bins = np.histogram(sampled.flatten(), 128, [0, 256])  
    
    non_zero_hist = hist[hist > 0]
    if len(non_zero_hist) > 0:
        min_hist_val = np.min(non_zero_hist)
        max_hist_val = np.max(hist)
        
        contrast_threshold = 0.1 * max_hist_val
        needs_contrast = min_hist_val < contrast_threshold
    else:
        needs_contrast = False
    
    needs_brightness = mean_brightness < 50 or mean_brightness > 150 or std_brightness < 20
    
    # Сохраняем в кэш
    result = (needs_brightness, needs_contrast, mean_brightness, std_brightness)
    _prev_analysis = result
    _prev_analysis_frame = img_array.copy()
    
    return result

def advanced_preprocessing(img):
    pil_img = img

    
    gray_img = pil_img.convert('L')
    img_array = np.array(gray_img)
    
    
    needs_brightness, needs_contrast, mean_brightness, std_brightness = analyze_histogram_needs(img_array)

    if needs_brightness:
        if mean_brightness < 60:
            # Для темных изображений
            target_brightness = 130
            brightness_factor = target_brightness / max(mean_brightness, 1)
            brightness_factor = min(brightness_factor, 2.5)  
        elif mean_brightness > 170:
            # Для слишком ярких изображений
            target_brightness = 140
            brightness_factor = target_brightness / mean_brightness
            brightness_factor = max(brightness_factor, 0.6)  
        else:
            brightness_factor = 1.1
            
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(brightness_factor)
    
    gray_img_processed = pil_img.convert('L')
    img_array_processed = np.array(gray_img_processed)
    if needs_contrast:
        
        
        
        
        p2, p98 = np.percentile(img_array_processed, [2, 98])
        
        if p98 > p2:  
            r, g, b = pil_img.split()
            r_array = np.array(r)
            g_array = np.array(g)
            b_array = np.array(b)
            
            
            scale = 255.0 / (p98 - p2)
            r_contrast = (r_array - p2) * scale
            g_contrast = (g_array - p2) * scale
            b_contrast = (b_array - p2) * scale
            
            r_contrast = np.clip(r_contrast, 0, 255).astype(np.uint8)
            g_contrast = np.clip(g_contrast, 0, 255).astype(np.uint8)
            b_contrast = np.clip(b_contrast, 0, 255).astype(np.uint8)
            
            result_img = Image.merge('RGB', 
                [Image.fromarray(r_contrast), 
                 Image.fromarray(g_contrast),
                 Image.fromarray(b_contrast)])
        else:
            result_img = pil_img
    else:
        result_img = pil_img

    return result_img

if __name__ == '__main__':
    cv2.namedWindow("result")
    #cv2.namedWindow("original")

    cap = cv2.VideoCapture(0)

    
    """
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    """
    prev_time = time.time()
    frame_count = 0
    fps_res = 0
    

    while True:
        flag, img = cap.read()
        fps_res = 0
        
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
            
            
            processed_img = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGB2BGR)
            
            
            if fps_res > 0:
                cv2.putText(img, f"FPS: {fps_res:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_img, f"FPS: {fps_res:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
                
            #cv2.imshow('original', img)
            cv2.imshow('result', processed_img)
            
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            cap.release()
            raise
 
        ch = cv2.waitKey(1)  
        if ch == 27:  # ESC
            break
        
    cap.release()
    cv2.destroyAllWindows()
