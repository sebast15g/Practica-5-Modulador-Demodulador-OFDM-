from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT
)
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout


class MplWidget(QWidget):
    """
    Widget completo: figura + toolbar (zoom, pan, save, reset)
    """
    def __init__(self, width=6, height=4, dpi=100):
        super().__init__()

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def clear(self):
        self.ax.clear()
        self.canvas.draw()
