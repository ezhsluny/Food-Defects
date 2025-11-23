import cv2
import time
import numpy as np
from PIL import Image, ImageEnhance
import os

# Кэш для хранения предыдущих результатов анализа
_prev_analysis_frame = None
_prev_analysis_frame_size = None

def analyze_histogram_needs(img_array):
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
    
    result = (needs_brightness, needs_contrast, mean_brightness, std_brightness)
    
    return result

def ensure_rgb_image(img):
    """Конвертирует изображение в RGB формат, если нужно"""
    if isinstance(img, Image.Image):
        if img.mode == 'RGBA':
            # Создаем белый фон и накладываем RGBA изображение
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])  # Используем альфа-канал как маску
            return background
        elif img.mode != 'RGB':
            return img.convert('RGB')
        else:
            return img
    else:
        # Если это numpy array
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                # RGBA to RGB
                return img[:, :, :3]
            elif img.shape[2] == 3:
                return img
        # Если это grayscale, конвертируем в RGB
        elif len(img.shape) == 2:
            return np.stack([img, img, img], axis=2)
        return img

def resize_to_match(img, target_size):
    """Изменяет размер изображения до целевого размера"""
    if img.size != target_size:
        return img.resize(target_size, Image.Resampling.LANCZOS)
    return img

def advanced_preprocessing(img):
    global _prev_analysis_frame, _prev_analysis_frame_size
    
    # Конвертируем в RGB если нужно
    pil_img = ensure_rgb_image(img)
    current_size = pil_img.size
    
    # Если кадр похож на предыдущий, используем кэш
    if _prev_analysis_frame is not None and _prev_analysis_frame_size == current_size:
        try:
            gray_img = pil_img.convert('L')
            img_array = np.array(gray_img)
            
            prev_gray = _prev_analysis_frame.convert('L')
            prev_array = np.array(prev_gray)
            
            # Проверяем, что размеры совпадают
            if img_array.shape == prev_array.shape:
                frame_diff = np.mean(np.abs(img_array - prev_array))
                if frame_diff < 5:  
                    return _prev_analysis_frame
        except Exception as e:
            print(f"Ошибка при сравнении кадров: {e}")
            # Продолжаем обычную обработку в случае ошибки
    
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
            # Разделяем каналы RGB
            r, g, b = pil_img.split()
            r_array = np.array(r)
            g_array = np.array(g)
            b_array = np.array(b)
            
            # Применяем контраст к каждому каналу
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

    # Сохраняем в кэш
    _prev_analysis_frame = result_img
    _prev_analysis_frame_size = current_size
    
    return result_img