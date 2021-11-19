"""
This is a minimal example that demonstrates usage of all different Qt GUI objects
that are available on this framework.
"""

import sys

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog

from gui.gui import GUI

class GUITest(GUI):
    def setupUi(self):
        #########################
        # Setup the Qt UI layout
        #########################
        # Resize the main window and set the stylesheet (CSS)
        MainWindow = self.mainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet("QMainWindow { background-color : rgb(54, 57, 63); color : rgb(220, 221, 222); }\n"
                                 "QLabel { background-color : rgb(54, 57, 63); color : rgb(220, 221, 222); }")

        # Add a graphics widget to the main window
        self.graphicsView = QtWidgets.QGraphicsView(self.mainWindow)
        self.graphicsView.setEnabled(True)
        self.graphicsView.setGeometry(QtCore.QRect(10, 10, 780, 580))
        self.graphicsView.setObjectName("graphicsView")

        # Zoom out a bit
        self.zoom = 40

        #########################
        # RMP Gui Zoo
        #########################
        self.add_text("RText", 0.0, 4.0, 1)
        self.add_text("This is some text", -2.0, 3.0, 1)

        self.add_text("RPolygon", 4.0, 4.0, 1)
        points = [(0.0, 0.0), (0.0, 2.0), (1.0, 1.0), (2.0, 2.0), (2.0, 0.0)]
        self.add_polygon(map(lambda p: (p[0]+4.0, p[1]+1.8), points), QtCore.Qt.transparent)

        self.add_text("RDisc", 8.0, 4.0, 1)
        self.add_disc(1.0, 8.0, 3.0)

        self.add_text("RDiscRobot", 11.0, 4.0, 1)
        self.add_disc_robot(1.0, 11.0, 3.0, "123", QtCore.Qt.red)

        self.add_text("RSegment", 0.0, 0.0, 1)
        self.add_segment(-1.0, -1.5, 0.83, 0.35 - 0.5)

        self.add_text("RSegment_angle", 4.0, 0.0, 1)
        self.add_segment_angle(3.7, -2.0, 1.5, 67.0 / 180.0 * 3.14152865)

        self.add_text("RCircleSegment", 9.0, 0.0, 1)
        self.add_circle_segment(1.0, 9.5, -1.5, 0.0, 70.0 / 180.0 * 3.14159265, False)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = GUITest()
    gui.set_program_name("RMP GUI Zoo")
    gui.mainWindow.show()
    sys.exit(app.exec_())
