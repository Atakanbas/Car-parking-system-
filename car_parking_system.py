import sys
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QMessageBox, QRubberBand, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QColor,QFileOpenEvent
from PyQt5.QtCore import Qt, QRect, QPoint, QSize, QTimer,QFile
from ultralytics import YOLO

class ParkingManagementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.parking_regions = []
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.model = YOLO("yolov8n_fold_0.pt")
        self.frame = None
        self.fps = 30
        self.drawing = False
        self.rubberBand = None
        self.origin = QPoint()
        self.webcam = False

    def initUI(self):
        self.setWindowTitle("Otopark Yönetim Sistemi")
        self.setGeometry(100, 100, 1400, 800)

        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)
        mainLayout = QHBoxLayout(mainWidget)

        leftPanel = QWidget(self)
        leftLayout = QVBoxLayout(leftPanel)
        mainLayout.addWidget(leftPanel, 1)

        self.label = QLabel(self)
        self.label.setGeometry(20, 20, 1000, 750)
        self.label.setStyleSheet("background-color: #222; border: 1px solid #333;")
        self.label.setMouseTracking(True)
        self.label.mousePressEvent = self.mouse_press_event
        self.label.mouseMoveEvent = self.mouse_move_event
        self.label.mouseReleaseEvent = self.mouse_release_event
        leftLayout.addWidget(self.label)

        rightPanel = QWidget(self)
        rightLayout = QVBoxLayout(rightPanel)
        mainLayout.addWidget(rightPanel, 0)

        self.btnLoadVideo = QPushButton("Video Yükle", self)
        self.btnLoadVideo.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btnLoadVideo.clicked.connect(self.load_video)
        rightLayout.addWidget(self.btnLoadVideo)

        self.btnWebcam = QPushButton("Web Kamerası", self)
        self.btnWebcam.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btnWebcam.clicked.connect(self.start_webcam)
        rightLayout.addWidget(self.btnWebcam)

        self.btnSelectFrame = QPushButton("Kare Seç", self)
        self.btnSelectFrame.setStyleSheet("background-color: #2196F3; color: white;")
        self.btnSelectFrame.clicked.connect(self.select_frame)
        rightLayout.addWidget(self.btnSelectFrame)

        self.btnSaveRegions = QPushButton("Alanları Kaydet", self)
        self.btnSaveRegions.setStyleSheet("background-color: #FFC107; color: white;")
        self.btnSaveRegions.clicked.connect(self.save_regions)
        rightLayout.addWidget(self.btnSaveRegions)

        self.btnStartDetection = QPushButton("Tespiti Başlat", self)
        self.btnStartDetection.setStyleSheet("background-color: #F44336; color: white;")
        self.btnStartDetection.clicked.connect(self.start_detection)
        rightLayout.addWidget(self.btnStartDetection)

        self.btnClearRegions = QPushButton("Alanları Temizle", self)
        self.btnClearRegions.setStyleSheet("background-color: #FF5722; color: white;")
        self.btnClearRegions.clicked.connect(self.clear_regions)
        rightLayout.addWidget(self.btnClearRegions)

        self.lblStatus = QLabel("Boş/Toplam: 0/0", self)
        self.lblStatus.setStyleSheet("font-weight: bold; color: #black;")
        rightLayout.addWidget(self.lblStatus)

        self.lblWarning = QLabel("", self)
        self.lblWarning.setStyleSheet("color: red; font-weight: bold;")
        rightLayout.addWidget(self.lblWarning)

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Video Yükle", "", "Video Dosyaları (*.mp4 *.avi)")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.webcam = False
            assert self.cap.isOpened(), "Video dosyası okunurken hata oluştu"
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            ret, self.frame = self.cap.read()
            if ret:
                self.display_image(self.frame)

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.webcam = True
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Uyarı", "Web kamerası açılamadı")
            return
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        ret, self.frame = self.cap.read()
        if ret:
            self.display_image(self.frame)
        self.timer.start(1000 // self.fps)

    def select_frame(self):
        if self.cap is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir video yükleyin")
            return
        ret, self.frame = self.cap.read()
        if ret:
            self.display_image(self.frame)

    def save_regions(self):
        if not self.parking_regions:
            QMessageBox.warning(self, "Uyarı", "Kaydedilecek park alanı yok")
            return
        regions = [{"id": idx, "points": region} for idx, region in enumerate(self.parking_regions)]
        with open("bounding_boxes.json", "w") as f:
            json.dump(regions, f)
        QMessageBox.information(self, "Bilgi", "Alanlar bounding_boxes.json dosyasına kaydedildi")

    def start_detection(self):
        if self.cap is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir video yükleyin")
            return
        if not self.parking_regions:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce park alanlarını seçip kaydedin")
            return
        self.timer.start(1000 // self.fps)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            return

        results = self.model(frame)
        with open("bounding_boxes.json") as f:
            parking_regions = json.load(f)

        occupied_regions = set()
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, clss):
                x1, y1, x2, y2 = map(int, box)

                if self.model.names[int(cls)] in ['car', 'truck', 'bus']:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                    for region_idx, region in enumerate(parking_regions):
                        points = np.array(region['points'], np.int32).reshape((-1, 1, 2))
                        if cv2.pointPolygonTest(points, ((x1 + x2) / 2, (y1 + y2) / 2), False) >= 0:
                            occupied_regions.add(region_idx)

        for region_idx, region in enumerate(parking_regions):
            points = np.array(region['points'], np.int32).reshape((-1, 1, 2))
            mask_color = (0, 255, 0) if region_idx not in occupied_regions else (0, 0, 255)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], mask_color)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        self.lblStatus.setText(f"Boş/Dolu Alanlar: {len(parking_regions) - len(occupied_regions)}/{len(parking_regions)}")
        self.display_image(frame)

    def display_image(self, img):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(qimg))
        self.label.setScaledContents(True)

    def mouse_press_event(self, event):
        if event.button() == Qt.LeftButton and self.frame is not None:
            self.origin = event.pos()
            if not self.rubberBand:
                self.rubberBand = QRubberBand(QRubberBand.Rectangle, self.label)
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()

    def mouse_move_event(self, event):
        if self.rubberBand:
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouse_release_event(self, event):
        if event.button() == Qt.LeftButton and self.rubberBand:
            rect = self.rubberBand.geometry()
            self.rubberBand.hide()
            if rect.isValid():
                x1, y1 = self.label_to_frame_coords(rect.topLeft())
                x2, y2 = self.label_to_frame_coords(rect.bottomRight())
                new_region = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                if not self.check_intersection(new_region):
                    self.parking_regions.append(new_region)
                    self.lblWarning.setText("")
                else:
                    self.lblWarning.setText("Kesişim var, lütfen yeniden deneyin!")
            self.update_display()

    def label_to_frame_coords(self, point):
        x = int(point.x() * self.frame.shape[1] / self.label.width())
        y = int(point.y() * self.frame.shape[0] / self.label.height())
        return x, y

    def check_intersection(self, new_region):
        new_rect = QRect(QPoint(new_region[0][0], new_region[0][1]), QPoint(new_region[2][0], new_region[2][1]))
        for region in self.parking_regions:
            existing_rect = QRect(QPoint(region[0][0], region[0][1]), QPoint(region[2][0], region[2][1]))
            if new_rect.intersects(existing_rect):
                return True
        return False

    def update_display(self):
        display_frame = self.frame.copy()
        for region in self.parking_regions:
            points = np.array(region, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        self.display_image(display_frame)

    def clear_regions(self):
        self.parking_regions = []
        if self.frame is not None:
            self.update_display()
        try:
            if QFile.exists("bounding_boxes.json"):
                QFile.remove("bounding_boxes.json")
        except Exception as e:
            QMessageBox.warning(self, "Uyarı", f"Dosya silinirken hata oluştu: {e}")
        self.lblStatus.setText("Boş/Dolu Alanlar: 0/0")
        QMessageBox.information(self, "Bilgi", "Tüm park alanları temizlendi")


    def stop_detection(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = ParkingManagementApp()
    mainWindow.show()
    sys.exit(app.exec_())
