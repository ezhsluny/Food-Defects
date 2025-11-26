import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

class DefectClassifier:
    def __init__(self, model_path='classificate_defect.onnx'):
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_info = self.session.get_inputs()[0]
            self.input_name = self.input_info.name
            self.output_name = self.session.get_outputs()[0].name
            
            self.classes = ['crit_def', 'mid_def', 'no_def', 'small_def']
            
            self.transform = transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            print(f"Ошибка инициализации модели: {e}")
            raise
    
    def preprocess_image(self, image):
        """Предобработка изображения с поддержкой разных форматов"""
        # Если передан путь к файлу
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Если передан объект PIL Image
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        
        # Если передан numpy array (от OpenCV)
        elif isinstance(image, np.ndarray):
            # Конвертируем из BGR в RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)  
        
        return image_tensor
    
    def predict(self, image):
        """Предсказание класса дефекта"""
        input_tensor = self.preprocess_image(image)
        input_numpy = input_tensor.numpy().astype(np.float32)
        input_numpy = np.transpose(input_numpy, (0, 2, 3, 1))

        outputs = self.session.run([self.output_name], {self.input_name: input_numpy})
        predictions = outputs[0]

        probabilities = torch.softmax(torch.tensor(predictions), dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = self.classes[predicted_class_idx]
        confidence = probabilities[0][predicted_class_idx].item()
        
        result = {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                self.classes[i]: probabilities[0][i].item() 
                for i in range(len(self.classes))
            }
        }
        
        return result

    def display_prediction(self, image, prediction_result, figsize=(12, 5)):
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
        
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax1.imshow(display_image)
        ax1.set_title(f'Предсказанный класс: {prediction_result["class"]}\nУверенность: {prediction_result["confidence"]:.4f}')
        ax1.axis('off')
        plt.tight_layout()
        plt.show()
    

if __name__ == "__main__":
    classifier = DefectClassifier('classificate_defect.onnx')
    
    # Тестирование с путем к файлу
    result1 = classifier.predict('val/crit_def/photo_14949.jpg')
    classifier.display_prediction('val/crit_def/photo_14949.jpg', result1)

    print("\nРезультат классификации:")
    print(f"Класс: {result1['class']}")
    print(f"Уверенность: {result1['confidence']:.4f}")
    print("Все вероятности:")
    for class_name, prob in result1['probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")
