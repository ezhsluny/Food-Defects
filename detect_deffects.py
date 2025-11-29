import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Union
from ultralytics import YOLO
import json

class AppleDefectDetector:
    """
    Класс для детекции дефектов яблок с использованием дообученной модели YOLOv8
    """
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.25):
        """
        Инициализация детектора
        
        Args:
            model_path: Путь к файлу весов модели (.pt)
            conf_threshold: Порог уверенности для детекции
        """
        self.conf_threshold = conf_threshold
        self.model = None
        self.class_names = ['crit_def', 'mid_def', 'small_def', 'no_def']
        self.class_colors = {
            'crit_def': (255, 0, 0),      # Красный - критический дефект
            'mid_def': (255, 165, 0),     # Оранжевый - значительный дефект
            'small_def': (255, 255, 0),   # Желтый - незначительный дефект
            'no_def': (0, 255, 0)         # Зеленый - нет дефекта
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("Предупреждение: Модель не загружена. Используйте load_model() для загрузки модели.")
    
    def load_model(self, model_path: str) -> bool:
        """
        Загрузка модели из файла
        
        Args:
            model_path: Путь к файлу весов модели
            
        Returns:
            bool: Успешность загрузки
        """
        try:
            self.model = YOLO(model_path)
            print(f"Модель успешно загружена из {model_path}")
            return True
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            return False
    
    def predict(self, 
                image: Union[str, np.ndarray, Image.Image],
                save: bool = False,
                save_dir: str = "results",
                return_image: bool = False) -> Dict:
        """
        Предсказание дефектов на изображении
        
        Args:
            image: Путь к изображению, numpy array или PIL Image
            save: Сохранять ли результат
            save_dir: Директория для сохранения результатов
            return_image: Возвращать ли изображение с визуализацией
            
        Returns:
            Dict: Результаты детекции
        """
        if self.model is None:
            raise ValueError("Модель не загружена. Сначала загрузите модель с помощью load_model()")
        
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            imgsz=320,
            verbose=False
        )
        
        detection_result = self._process_results(results[0])

        if save:
            self._save_results(detection_result, save_dir)
        
        if return_image:
            vis_image = self._visualize_detections(image, detection_result)
            detection_result['visualization'] = vis_image
        
        return detection_result
    
    def _process_results(self, result) -> Dict:
        """
        Обработка сырых результатов YOLO
        
        Args:
            result: Результат предсказания YOLO
            
        Returns:
            Dict: Структурированные результаты
        """

        boxes = result.boxes
        detection_result = {
            'original_image_shape': result.orig_shape,
            'detections': [],
            'summary': {
                'total_detections': 0,
                'defects_count': 0,
                'defects_by_class': {cls_name: 0 for cls_name in self.class_names}
            }
        }
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:

                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                xywh = box.xywh[0].cpu().numpy()  # [x_center, y_center, width, height]
                
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]
                
                detection = {
                    'bbox': {
                        'xyxy': xyxy.tolist(),
                        'xywh': xywh.tolist(),
                        'area': (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                    },
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': float(confidence)
                }
                
                detection_result['detections'].append(detection)
                
                detection_result['summary']['total_detections'] += 1
                if class_name != 'no_def':
                    detection_result['summary']['defects_count'] += 1
                detection_result['summary']['defects_by_class'][class_name] += 1
        
        return detection_result
    
    def _visualize_detections(self, 
                            image: Union[str, np.ndarray, Image.Image],
                            detection_result: Dict) -> np.ndarray:
        """
        Визуализация детекций на изображении
        
        Args:
            image: Исходное изображение
            detection_result: Результаты детекции
            
        Returns:
            np.ndarray: Изображение с визуализацией
        """

        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        for detection in detection_result['detections']:
            bbox = detection['bbox']['xyxy']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            color = self.class_colors.get(class_name, (255, 255, 255))
            
            x1, y1, x2, y2 = map(int, bbox)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)        
        return img
    
    def _save_results(self, detection_result: Dict, save_dir: str):
        """
        Сохранение результатов детекции
        
        Args:
            detection_result: Результаты детекции
            save_dir: Директория для сохранения
        """
        os.makedirs(save_dir, exist_ok=True)
        
        json_path = os.path.join(save_dir, "detection_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json_ready_result = json.loads(json.dumps(detection_result, default=str))
            json.dump(json_ready_result, f, indent=2, ensure_ascii=False)
        
        print(f"Результаты сохранены в {json_path}")
    
    def get_detection_summary(self, detection_result: Dict) -> str:
        """
        Получение текстовой сводки по результатам детекции
        
        Args:
            detection_result: Результаты детекции
            
        Returns:
            str: Текстовая сводка
        """
        summary = detection_result['summary']
        
        report = f"""
        СВОДКА ДЕТЕКЦИИ ДЕФЕКТОВ:
        -------------------------
        Всего обнаружено объектов: {summary['total_detections']}
        Всего дефектов: {summary['defects_count']}
        
        Распределение по классам:
        """
        
        for class_name, count in summary['defects_by_class'].items():
            report += f"  - {class_name}: {count}\n"
        
        if summary['defects_by_class']['crit_def'] > 0:
            report += "\n⚠️ ВНИМАНИЕ: Обнаружены критические дефекты!\n"
        elif summary['defects_by_class']['mid_def'] > 0:
            report += "\nℹ️ Обнаружены значительные дефекты\n"
        elif summary['defects_by_class']['small_def'] > 0:
            report += "\n✅ Обнаружены незначительные дефекты\n"
        else:
            report += "\n✅ Дефекты не обнаружены\n"
        
        return report
    
    def set_confidence_threshold(self, threshold: float):
        """
        Установка порога уверенности
        
        Args:
            threshold: Новый порог уверенности (0.0-1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.conf_threshold = threshold
            print(f"Порог уверенности установлен: {threshold}")
        else:
            print("Порог уверенности должен быть между 0.0 и 1.0")

def create_detector(model_path: str = "best.pt", conf_threshold: float = 0.25) -> AppleDefectDetector:
    """
    Фабричная функция для создания детектора
    
    Args:
        model_path: Путь к модели
        conf_threshold: Порог уверенности
        
    Returns:
        AppleDefectDetector: Инициализированный детектор
    """
    return AppleDefectDetector(model_path, conf_threshold)

if __name__ == "__main__":

    detector = AppleDefectDetector("apples_defects_yolov8.pt")
    
    result = detector.predict("test1.jpg", save=True, return_image=True)

    print(detector.get_detection_summary(result))

    if 'visualization' in result:
        plt.figure(figsize=(12, 8))
        plt.imshow(result['visualization'])
        plt.axis('off')
        plt.title("Результаты детекции дефектов яблок")
        plt.show()