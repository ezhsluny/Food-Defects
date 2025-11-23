// Элементы DOM
const video = document.getElementById('video');
const startCameraBtn = document.getElementById('start-camera');
const startProcessingBtn = document.getElementById('start-processing');
const stopProcessingBtn = document.getElementById('stop-processing');
const stopCameraBtn = document.getElementById('stop-camera');
const cameraStatus = document.getElementById('camera-status');
const cameraResultCanvas = document.getElementById('camera-result-canvas');
const cameraNoResult = document.getElementById('camera-no-result');

// Элементы статистики
const fpsCounter = document.getElementById('fps-counter');
const applesCounter = document.getElementById('apples-counter');
const defectsCounter = document.getElementById('defects-counter');
const processingStatus = document.getElementById('processing-status');

// Элементы для загрузки видео
const uploadedVideo = document.getElementById('uploaded-video');
const processedVideo = document.getElementById('processed-video');
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const processUploadedBtn = document.getElementById('process-uploaded');
const downloadVideoBtn = document.getElementById('download-video');
const uploadStatus = document.getElementById('upload-status');
const uploadResults = document.getElementById('upload-results');
const uploadNoResult = document.getElementById('upload-no-result');

// Переменные для реального времени
let stream = null;
let isProcessing = false;
let processingInterval = null;
let frameCount = 0;
let lastFpsUpdate = 0;
let currentFps = 0;
let lastApplesCount = 0;
let lastDefectsCount = 0;

// Canvas context
const canvasCtx = cameraResultCanvas.getContext('2d');

// Управление вкладками
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        tab.classList.add('active');
        const tabId = `${tab.dataset.tab}-tab`;
        const tabContent = document.getElementById(tabId);
        if (tabContent) {
            tabContent.classList.add('active');
        }
        
        // Останавливаем обработку при переключении вкладок
        if (tab.dataset.tab !== 'camera' && isProcessing) {
            stopRealtimeProcessing();
        }
    });
});

// Работа с камерой
if (startCameraBtn) {
    startCameraBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30 }
                } 
            });
            video.srcObject = stream;
            
            // Настраиваем canvas
            cameraResultCanvas.width = 640;
            cameraResultCanvas.height = 480;
            
            startCameraBtn.disabled = true;
            stopCameraBtn.disabled = false;
            startProcessingBtn.disabled = false;
            
            showStatus(cameraStatus, 'Камера активирована. Запустите обработку.', 'success');
            updateProcessingStatus('Готов к работе');
            
        } catch (err) {
            showStatus(cameraStatus, `Ошибка доступа к камере: ${err.message}`, 'error');
        }
    });
}

if (stopCameraBtn) {
    stopCameraBtn.addEventListener('click', () => {
        stopRealtimeProcessing();
        
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            stream = null;
        }
        
        startCameraBtn.disabled = false;
        stopCameraBtn.disabled = true;
        startProcessingBtn.disabled = true;
        stopProcessingBtn.disabled = true;
        
        // Очищаем canvas
        canvasCtx.clearRect(0, 0, cameraResultCanvas.width, cameraResultCanvas.height);
        cameraNoResult.classList.remove('hidden');
        
        showStatus(cameraStatus, 'Камера выключена', 'info');
        updateProcessingStatus('Остановлено');
        resetStats();
    });
}

// Запуск обработки в реальном времени
if (startProcessingBtn) {
    startProcessingBtn.addEventListener('click', () => {
        startRealtimeProcessing();
    });
}

// Остановка обработки
if (stopProcessingBtn) {
    stopProcessingBtn.addEventListener('click', () => {
        stopRealtimeProcessing();
    });
}

function startRealtimeProcessing() {
    if (!stream || isProcessing) return;
    
    isProcessing = true;
    startProcessingBtn.disabled = true;
    stopProcessingBtn.disabled = false;
    
    // Показываем canvas
    cameraNoResult.classList.add('hidden');
    
    // Запускаем обработку кадров
    processingInterval = setInterval(processVideoFrame, 100); // 10 FPS для обработки
    
    showStatus(cameraStatus, 'Обработка запущена', 'success');
    updateProcessingStatus('Обрабатывается...');
    processingStatus.classList.add('processing-active');
}

function stopRealtimeProcessing() {
    isProcessing = false;
    startProcessingBtn.disabled = false;
    stopProcessingBtn.disabled = true;
    
    if (processingInterval) {
        clearInterval(processingInterval);
        processingInterval = null;
    }
    
    showStatus(cameraStatus, 'Обработка остановлена', 'info');
    updateProcessingStatus('Остановлено');
    processingStatus.classList.remove('processing-active');
}

async function processVideoFrame() {
    if (!isProcessing) return;
    
    try {
        // Создаем временный canvas для захвата кадра
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        tempCtx.drawImage(video, 0, 0);
        
        // Конвертируем в Blob
        tempCanvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');
            formData.append('realtime', 'true');
            
            const response = await fetch('/process-realtime', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                displayRealtimeResult(result);
            }
        }, 'image/jpeg', 0.8);
        
        // Обновляем FPS
        updateFps();
        
    } catch (error) {
        console.error('Error processing frame:', error);
    }
}

function displayRealtimeResult(data) {
    if (!data || data.error) return;
    
    // Обновляем статистику
    if (data.apples_count !== undefined) {
        lastApplesCount = data.apples_count;
        applesCounter.textContent = lastApplesCount;
    }
    
    if (data.defects_count !== undefined) {
        lastDefectsCount = data.defects_count;
        defectsCounter.textContent = lastDefectsCount;
    }
    
    // Отображаем обработанное изображение
    if (data.processed_image) {
        const img = new Image();
        img.onload = function() {
            // Очищаем canvas
            canvasCtx.clearRect(0, 0, cameraResultCanvas.width, cameraResultCanvas.height);
            
            // Рисуем изображение
            canvasCtx.drawImage(img, 0, 0, cameraResultCanvas.width, cameraResultCanvas.height);
            
            // Добавляем статистику поверх изображения
            canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            canvasCtx.fillRect(10, 10, 200, 80);
            
            canvasCtx.fillStyle = 'white';
            canvasCtx.font = '14px Arial';
            canvasCtx.fillText(`Яблок: ${lastApplesCount}`, 20, 30);
            canvasCtx.fillText(`Дефекты: ${lastDefectsCount}`, 20, 50);
            canvasCtx.fillText(`FPS: ${currentFps}`, 20, 70);
            
            if (data.classification_results) {
                const cls = data.classification_results;
                canvasCtx.fillText(`Класс: ${cls.class}`, 20, 90);
            }
        };
        img.src = `data:image/jpeg;base64,${data.processed_image}`;
    }
}

function updateFps() {
    frameCount++;
    const now = performance.now();
    
    if (now - lastFpsUpdate >= 1000) {
        currentFps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
        fpsCounter.textContent = currentFps;
        frameCount = 0;
        lastFpsUpdate = now;
    }
}

function updateProcessingStatus(status) {
    if (processingStatus) {
        processingStatus.textContent = status;
    }
}

function resetStats() {
    fpsCounter.textContent = '0';
    applesCounter.textContent = '0';
    defectsCounter.textContent = '0';
    currentFps = 0;
    frameCount = 0;
    lastApplesCount = 0;
    lastDefectsCount = 0;
}

// Загрузка файлов (остается без изменений)
if (uploadArea && fileInput) {
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

function handleFileUpload(file) {
    if (!file.type.startsWith('video/') && !file.type.startsWith('image/')) {
        showStatus(uploadStatus, 'Пожалуйста, выберите видео или изображение', 'error');
        return;
    }

    const url = URL.createObjectURL(file);
    
    if (file.type.startsWith('video/') && uploadedVideo) {
        uploadedVideo.src = url;
        uploadedVideo.classList.remove('hidden');
        
        // Сбрасываем обработанное видео
        if (processedVideo) {
            processedVideo.classList.add('hidden');
            processedVideoBlob = null;
        }
        if (uploadNoResult) {
            uploadNoResult.classList.remove('hidden');
        }
        if (downloadVideoBtn) {
            downloadVideoBtn.disabled = true;
        }
    }
    
    if (processUploadedBtn) processUploadedBtn.disabled = false;
    
    showStatus(uploadStatus, `Файл "${file.name}" загружен успешно`, 'success');
}

// Обработка загруженного видео
if (processUploadedBtn) {
    processUploadedBtn.addEventListener('click', async () => {
        if (!fileInput || !fileInput.files.length) return;
        
        const file = fileInput.files[0];
        const isVideo = file.type.startsWith('video/');
        
        showStatus(uploadStatus, `Обработка ${isVideo ? 'видео' : 'изображения'}...`, 'info');
        
        try {
            const formData = new FormData();
            formData.append(isVideo ? 'video' : 'image', file);
            
            const endpoint = isVideo ? '/process-video' : '/process-image';
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                
                if (isVideo && result.processed_video) {
                    displayProcessedVideo(result);
                }
                
                displayResults(result, uploadResults);
                showStatus(uploadStatus, 'Обработка завершена', 'success');
            } else {
                const error = await response.json();
                throw new Error(error.error || 'Ошибка сервера');
            }
        } catch (error) {
            showStatus(uploadStatus, `Ошибка обработки: ${error.message}`, 'error');
        }
    });
}

// Функция для отображения обработанного видео
function displayProcessedVideo(data) {
    if (!processedVideo || !uploadNoResult) return;
    
    if (data.processed_video) {
        const videoBlob = base64ToBlob(data.processed_video, 'video/mp4');
        processedVideoBlob = videoBlob;
        
        const videoUrl = URL.createObjectURL(videoBlob);
        processedVideo.src = videoUrl;
        processedVideo.classList.remove('hidden');
        uploadNoResult.classList.add('hidden');
        
        if (downloadVideoBtn) {
            downloadVideoBtn.disabled = false;
        }
    }
}

// Функция для конвертации base64 в Blob
function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteArrays = [];
    
    for (let offset = 0; offset < byteCharacters.length; offset += 512) {
        const slice = byteCharacters.slice(offset, offset + 512);
        const byteNumbers = new Array(slice.length);
        
        for (let i = 0; i < slice.length; i++) {
            byteNumbers[i] = slice.charCodeAt(i);
        }
        
        const byteArray = new Uint8Array(byteNumbers);
        byteArrays.push(byteArray);
    }
    
    return new Blob(byteArrays, { type: mimeType });
}

// Скачивание обработанного видео
if (downloadVideoBtn) {
    downloadVideoBtn.addEventListener('click', () => {
        if (!processedVideoBlob) return;
        
        const url = URL.createObjectURL(processedVideoBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'processed_video.mp4';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showStatus(uploadStatus, 'Видео сохранено', 'success');
    });
}

// Вспомогательные функции
function showStatus(element, message, type) {
    if (!element) return;
    element.textContent = message;
    element.className = 'status';
    if (type) {
        element.classList.add(type);
    }
}

function displayResults(data, container) {
    if (!container) return;
    
    container.innerHTML = '';
    
    if (data.error) {
        container.innerHTML = `<div class="status error">${data.error}</div>`;
        return;
    }
    
    let html = '<h3>Результаты анализа:</h3>';
    
    if (data.detection_results) {
        html += `<div class="result-item">Детекция: ${data.detection_results}</div>`;
    }
    
    if (data.classification_results) {
        const cls = data.classification_results;
        html += `<div class="result-item">Классификация: ${cls.class} (${(cls.confidence * 100).toFixed(1)}%)</div>`;
    }
    
    if (data.apples_count !== undefined) {
        html += `<div class="result-item">Найдено яблок: ${data.apples_count}</div>`;
    }
    
    if (data.defects_count !== undefined) {
        html += `<div class="result-item">Дефектов обнаружено: ${data.defects_count}</div>`;
    }
    
    if (data.frames_processed !== undefined) {
        html += `<div class="result-item">Обработано кадров: ${data.frames_processed}</div>`;
    }
    
    if (data.processing_time !== undefined) {
        html += `<div class="result-item">Время обработки: ${data.processing_time.toFixed(2)} сек</div>`;
    }
    
    container.innerHTML = html;
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    console.log('Страница загружена, система реального времени готова');
});