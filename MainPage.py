import os
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtCore import Qt, QMargins
from TireSelectionPage import TireSelectionPage



class MainPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Tire Performance Post-Processing Tool')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #181c36;")

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 350, 0, 0))
        self.setLayout(layout)

        label = QLabel('Tire Performance Post-Processing Tool', self)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: white; font-family: agencyfb; font-size: 60px;")
        layout.addWidget(label)

        subtitle = QLabel('Analyze and visualize tire performance data effortlessly', self)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #FCA306; font-family: agencyfb; font-size: 22px; font-weight: thin;")
        layout.addWidget(subtitle)

        load_button = QPushButton('Load Data', self)
        load_button.setStyleSheet("""
            QPushButton {
                background-color: #333646;
                color: #e2e8f0;
                font-family: agencyfb;
                font-size: 20px;
                border: none;
                padding: 10px;
                padding-left: 0px;
                padding-right: 0px;
                border-radius: 0px;
            }
            QPushButton:pressed {
                background-color: #e2e8f0;
                color: #333646;
            }
        """)
        load_button.clicked.connect(self.openFileDialog)
        layout.addWidget(load_button)

        layout.addStretch()

    def openFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", options=options)
        if directory:
            self.processDirectory(directory)

    def processDirectory(self, directory):
        self.new_page = TireSelectionPage(directory)
        self.new_page.show()
        self.close()

