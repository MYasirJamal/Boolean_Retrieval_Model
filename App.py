#====================================================================================
#                                                                                   #
#                                                                                   #
#                               UTILITIES                                           #
#                                                                                   #
#                                                                                   #
#====================================================================================

import sys
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

import Assign1


#====================================================================================
#                                                                                   #
#                                                                                   #
#                                  MAIN WINDOW                                      #
#                                                                                   #
#                                                                                   #
#====================================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window properties
        self.setWindowTitle('Search Engine')
        self.setGeometry(300, 300, 700, 300)

        # Set background color
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        self.setPalette(palette)

        # Create a widget to hold the labels and line edits
        widget = QWidget()
        self.setCentralWidget(widget)

        # Create a vertical layout for the widget
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Create a label and line edit for the query input
        query_layout = QHBoxLayout()
        layout.addLayout(query_layout)

        self.query_label = QLabel('Enter a query:')
        self.query_label.setStyleSheet('color: white')
        query_layout.addWidget(self.query_label)

        self.query_edit = QLineEdit()
        self.query_edit.setStyleSheet('background-color: white')
        query_layout.addWidget(self.query_edit)

        # Create a button to process the query
        self.query_button = QPushButton('Process Query')
        self.query_button.setStyleSheet('color: black')
        self.query_button.clicked.connect(self.push_button)
        layout.addWidget(self.query_button)

        # Create a label to display result
        self.result_label = QLabel()
        self.result_label.setStyleSheet('color: white')
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

    def push_button(self):
        
        query = self.query_edit.text()
        result, time_taken = Assign1.processQuery(query)
        time_taken = str(time_taken*1000)

        if len(result) == 0:
            self.result_label.setText("Result: No results found\nTime taken: "+time_taken+" ms")
        else:
            self.result_label.setText("Result: "+str(result)+"\nTime taken: "+time_taken+" ms")
        

if __name__ == '__main__':
    
    #process docs at the start of application
    Assign1.processDocs()

    #create GUI window
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
