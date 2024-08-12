import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QComboBox, QTreeWidget, QTreeWidgetItem, QPushButton
from TireProcessingPage import TireProcessingPage


class TireSelectionPage(QWidget):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Sélectionner le Pneu et le Mode')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #181c36; color: white;")

        layout = QVBoxLayout()

        self.mode_selector = QComboBox(self)
        self.mode_selector.addItems(["Braking", "R&D"])
        layout.addWidget(self.mode_selector)

        self.tire_tree = QTreeWidget()
        self.tire_tree.setHeaderLabel('Pneus')
        self.tire_tree.setStyleSheet("""
            QTreeWidget {
                background-color: #2e3440;
                color: white;
                font-family: agencyfb;
                font-size: 10px;
                border: none;
            }
            QTreeWidget::item {
                height: 25px;
                padding: 5px;
            }
            QTreeWidget::item:selected {
                background-color: #4c566a;
            }
            QTreeWidget::item:hover {
                background-color: #434c5e;
            }
        """)
        layout.addWidget(self.tire_tree)

        self.loadTires(self.directory)

        proceed_button = QPushButton('Procéder', self)
        proceed_button.setStyleSheet("background-color: #4c566a; color: white; font-size: 16px;")
        proceed_button.clicked.connect(self.proceedToProcessing)
        layout.addWidget(proceed_button)

        self.setLayout(layout)

    def loadTires(self, directory):
        for tire in os.listdir(directory):
            tire_path = os.path.join(directory, tire)
            if os.path.isdir(tire_path):
                tire_item = QTreeWidgetItem(self.tire_tree, [tire])
                self.tire_tree.addTopLevelItem(tire_item)
                for mode in ["Braking", "R&D"]:
                    mode_path = os.path.join(tire_path, mode)
                    if os.path.isdir(mode_path):
                        mode_item = QTreeWidgetItem(tire_item, [mode])
                        tire_item.addChild(mode_item)
                        for fz in ["FZ40", "FZ80", "FZ120"]:
                            fz_path = os.path.join(mode_path, fz)
                            if os.path.isdir(fz_path):
                                fz_item = QTreeWidgetItem(mode_item, [fz])
                                mode_item.addChild(fz_item)
                                for condition in ["ND", "NW"]:
                                    condition_path = os.path.join(fz_path, condition)
                                    if os.path.isdir(condition_path):
                                        condition_item = QTreeWidgetItem(fz_item, [condition])
                                        fz_item.addChild(condition_item)

    def proceedToProcessing(self):
        selected_item = self.tire_tree.currentItem()
        if selected_item:
            tire_name = selected_item.parent().text(0) if selected_item.parent() else selected_item.text(0)
            mode_name = selected_item.text(0) if selected_item.parent() else None
            if mode_name:
                tire_path = os.path.join(self.directory, tire_name, mode_name)
            else:
                tire_path = os.path.join(self.directory, tire_name, self.mode_selector.currentText())
            self.processing_page = TireProcessingPage(tire_path, self)
            self.processing_page.show()