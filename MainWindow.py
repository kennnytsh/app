from PyQt5.QtWidgets import QApplication, QMainWindow, QSplitter
from TireProcessingPage import TireProcessingPage
from TireAnalysis import TireAnalysis

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Tire Analysis and Processing')
        self.setGeometry(100, 100, 1600, 900)

        splitter = QSplitter()

        self.tire_processing = TireProcessingPage()
        self.tire_analysis = TireAnalysis()  # Passez les arguments nécessaires

        splitter.addWidget(self.tire_processing)
        splitter.addWidget(self.tire_analysis)

        self.setCentralWidget(splitter)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())