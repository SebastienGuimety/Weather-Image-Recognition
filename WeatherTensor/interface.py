from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QProgressBar
from PyQt5.QtGui import QPixmap, QImageReader
from PyQt5.QtCore import Qt, QTimer
from WeatherClassTest import WeatherClassifier  # Make sure the file name is correct
import sys

class WeatherApp(QWidget):
    def __init__(self):
        super().__init__()

        self.weather_classifier = WeatherClassifier()
        self.current_image_path = None  # Initialize the variable
        self.image_size = (500, 300)  # Set the desired size for displayed images

        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 600, 400)
        self.setWindowTitle('Weather App')

        self.lbl_img = QLabel(self)
        self.lbl_img.setGeometry(50, 20, *self.image_size)
        self.lbl_img.setAlignment(Qt.AlignCenter)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(50, 340, 230, 30)

        btn_load = QPushButton('Load Image', self)
        btn_load.setGeometry(300, 340, 100, 30)
        btn_load.clicked.connect(self.loadImage)

        btn_run_model = QPushButton('Run Model', self)
        btn_run_model.setGeometry(430, 340, 100, 30)
        btn_run_model.clicked.connect(self.runModel)

        self.result_label = QLabel(self)
        self.result_label.setGeometry(50, 380, 500, 20)

        vbox = QVBoxLayout()
        vbox.addWidget(self.lbl_img)
        vbox.addWidget(self.progress_bar)
        vbox.addWidget(btn_load)
        vbox.addWidget(btn_run_model)
        vbox.addWidget(self.result_label)

        self.setLayout(vbox)

        self.show()

    def loadImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.current_image_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp *.gif *.jpeg);;All Files (*)", options=options)
        if self.current_image_path:
            self.displayImage(self.current_image_path)

    def runModel(self):
        if self.current_image_path:
            self.progress_bar.setValue(0)  # Reset progress bar
            self.progress_bar.setRange(0, 0)  # Set the progress bar to indeterminate mode

            # Run the model asynchronously (you can replace this with your actual model processing)
            QTimer.singleShot(2000, self.processModel)  # Simulating a 2-second delay

    def processModel(self):
        prediction = self.weather_classifier.predict_weather(self.current_image_path)
        self.progress_bar.setRange(0, 1)  # Reset the progress bar range
        self.progress_bar.setValue(1)  # Set the progress bar to completion

        self.result_label.setText(f"Predicted weather class: {prediction}")

    def displayImage(self, file_path):
        # Resize the image to the desired size
        pixmap = QPixmap(file_path).scaled(*self.image_size, Qt.KeepAspectRatio)
        self.lbl_img.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    weather_app = WeatherApp()
    sys.exit(app.exec_())
