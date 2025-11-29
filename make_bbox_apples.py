from ultralytics import YOLO
from PIL import Image
import numpy as np

class DetectorApples():

    def __init__(self):
        self.model = YOLO("yolo11n.pt")  
        self.apple_class_id = self._get_apple_class_id()

    def _get_apple_class_id(self):
        """Определяет ID класса 'apple' в модели"""

        class_names = self.model.names
        apple_variants = ['apple', 'Apple', 'APPLES', 'apples']
        
        for class_id, class_name in class_names.items():
            if class_name.lower() in [variant.lower() for variant in apple_variants]:
                print(f"Найден класс яблок: '{class_name}' с ID: {class_id}")
                return class_id
        print("Предупреждение: класс 'apple' не найден в модели. Используется класс с ID 0")
        return 0

    def crop_bboxes(self, image):
        """Вырезает регионы только с обнаруженными яблоками из изображения"""

        cropped_images = []
        bbox_info = []

        results = self.model(image)  
        
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        

        for result in results:
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):

                    cls = int(box.cls[0].cpu().numpy())
                    
                    if cls != self.apple_class_id:
                        continue
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)

                    padding = 5
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(image_np.shape[1], x2 + padding)
                    y2 = min(image_np.shape[0], y2 + padding)
                    
                    cropped_region = image_np[y1:y2, x1:x2]
                    
                    conf = box.conf[0].cpu().numpy()
                    
                    cropped_images.append(cropped_region)
                    bbox_info.append({
                        'coordinates': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': cls,
                        'class_name': self.model.names[cls]
                    })
        
        return cropped_images, bbox_info