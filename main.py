import sys
from PyQt5.QtWidgets import QApplication
from MainPage import MainPage
from MainWindow import MainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainPage()
    main_window.show()
    sys.exit(app.exec_())
