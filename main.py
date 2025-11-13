from detect_deffects import AppleDefectDetector
from classification_defects import DefectClassifier
from video_capture import advanced_preprocessing

import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageEnhance
import time
import numpy as np

def display_pred_on_capture(image, prediction_result):
    """Отображение изображения с предсказанием"""
    # Если передан путь к файлу
    if isinstance(image, str):
        display_image = Image.open(image)
    # Если передан объект PIL Image
    elif isinstance(image, Image.Image):
        display_image = image
    # Если передан numpy array
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        display_image = Image.fromarray(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Конвертируем обратно в numpy array для OpenCV
    display_image_np = np.array(display_image)
    # Конвертируем RGB в BGR для OpenCV
    if len(display_image_np.shape) == 3 and display_image_np.shape[2] == 3:
        display_image_np = cv2.cvtColor(display_image_np, cv2.COLOR_RGB2BGR)
    
    return display_image_np

if __name__ == '__main__':

    cv2.namedWindow("result")

    cap = cv2.VideoCapture('apples.mp4')
    detectorDefect = AppleDefectDetector("apples_defects_yolov8.pt")
    classifier = DefectClassifier('classificate_defect.onnx')


    video = int(input())

    if video:

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        """
        
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
                
                
                #image = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGB2BGR)
                image = processed_pil
                
                result1 = classifier.predict(image)
                
                processed_img = display_pred_on_capture(image, result1)

                print("\nРезультат классификации:")
                print(f"Класс: {result1['class']}")
                print(f"Уверенность: {result1['confidence']:.4f}")
                print("Все вероятности:")
                for class_name, prob in result1['probabilities'].items():
                    print(f"  {class_name}: {prob:.4f}")

                
                result = detectorDefect.predict(image, save=True, return_image=True)

                print(detectorDefect.get_detection_summary(result))

                processed_img = cv2.cvtColor(result['visualization'], cv2.COLOR_RGB2BGR)

                if fps_res > 0:
                    cv2.putText(img, f"FPS: {fps_res:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(processed_img, f"FPS: {fps_res:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
    else:
        image = 'val/crit_def/photo_14949.jpg'

        result1 = classifier.predict(image)
        

        print("\nРезультат классификации:")
        print(f"Класс: {result1['class']}")
        print(f"Уверенность: {result1['confidence']:.4f}")
        print("Все вероятности:")
        for class_name, prob in result1['probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
        
        # Детекция на одном изображении
        result = detectorDefect.predict(image, save=True, return_image=True)
        
        # Вывод сводки
        print(detectorDefect.get_detection_summary(result))

        
        result_image = cv2.cvtColor(result['visualization'], cv2.COLOR_BGR2RGB)
        classifier.display_prediction(result_image, result1)


