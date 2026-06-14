from arpes_tools import mpes
from arpes_tools.loader import DataLoader
from gui.fake_data import make_fake_trarpes_data

import sys
import os
import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QCheckBox,
    QPushButton,
    QFrame,
    QMenu,
    QAction,
    QFileDialog
)

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMenu
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class BasePlot:
    def __init__(self, ax, state):
        self.ax = ax
        self.state = state

    def setup_axis(self):
        pass
    
    def attach(self, ax):
        self.ax = ax

    def update(self):
        raise NotImplementedError

class MomentumMapPlot(BasePlot):

    def __init__(self, ax, state):
        super().__init__(ax, state)
        _, _, self.im, = mpes.plot_momentum_maps(
            self.state.I,
            self.state.E, self.state.E_int,
            self.state.delay, self.state.delay_int,
            fig = self.ax.figure, ax=ax,
        )

        self.vline = self.ax.axvline(
            self.state.kx,
            color="maroon",
            ls="--"
        )

        self.hline = self.ax.axhline(
            self.state.ky,
            color="purple",
            ls="--"
        )

    def setup_axis(self):
        self.ax.set_aspect("equal")
        self.im.set_clim(0, 1)

    def update(self):
        frame = mpes.get_momentum_map(
            self.state.I,
            self.state.E, self.state.E_int,
            self.state.delay, self.state.delay_int,
            norm = self.state.norm
        )

        self.im.set_data(frame)
        self.ax.set_title(f"{self.state.E:.2f} eV")

        self.vline.set_xdata([self.state.kx, self.state.kx])
        self.hline.set_ydata([self.state.ky, self.state.ky])

class EDCPlot(BasePlot):

    def __init__(self, ax, state):
        super().__init__(ax, state)   

        _, _, self.line, = mpes.plot_edc(
            self.state.I,
            (self.state.kx, self.state.ky),
            (self.state.k_int, self.state.k_int),
            norm_trace = self.state.norm,
            subtract_neg = self.state.subtract_neg,
            fig = self.ax.figure, ax=ax
        )

    def setup_axis(self):
        self.ax.set_aspect("auto")
        self.ax.set_ylim(0, 1)

    def update(self):
        edc = mpes.get_edc(
            self.state.I,
            (self.state.kx, self.state.ky),
            (self.state.k_int, self.state.k_int),
            norm_trace = self.state.norm,
            subtract_neg = self.state.subtract_neg
        )

        self.line.set_xdata(self.state.I.E.values)
        self.line.set_ydata(edc)
        #self.line.set_data(self.state.I.E.values, edc)

class TimeTracePlot(BasePlot):

    def __init__(self, ax, state):
        super().__init__(ax, state)

        _, _, self.line, = mpes.plot_time_traces(
            self.state.I,
            self.state.E, self.state.E_int,
            (self.state.kx, self.state.ky),
            (self.state.k_int, self.state.k_int),
            norm_trace = self.state.norm,
            subtract_neg = self.state.subtract_neg,
            fig = self.ax.figure, ax=ax
        )
        #ax.legend('off')

    def setup_axis(self):
        self.ax.set_aspect("auto")
        self.ax.set_ylim(0,1)

    def update(self):
        trace = mpes.get_time_trace(
            self.state.I,
            self.state.E, self.state.E_int,
            (self.state.kx, self.state.ky),
            (self.state.k_int, self.state.k_int),
            norm_trace = self.state.norm,
            subtract_neg = self.state.subtract_neg
        )
        #trace = trace / trace.max()
        self.line.set_xdata(self.state.I.delay.values)
        self.line.set_ydata(trace)

class WaterfallPlot(BasePlot):

    def __init__(self, ax, state):
        super().__init__(ax, state)
        _, _, self.im, = mpes.plot_waterfall(
            self.state.I,
            self.state.kx, self.state.k_int,
            self.state.ky, self.state.k_int,
            subtract_neg = self.state.subtract_neg,
            fig = self.ax.figure, ax=ax,
            E_enhance = 1        )

    def setup_axis(self):
        self.ax.set_aspect("auto")
        self.im.set_clim(0, 1)
        self.ax.set_ylim(1,3)
        
    def update(self):
        frame = mpes.get_waterfall(
            self.state.I,
            self.state.kx, self.state.k_int,
            self.state.ky, self.state.k_int
        )
        #frame = frame / frame.max()

        self.im.set_data(frame)

class State:
    def __init__(self):
        self.I = make_fake_trarpes_data()
        self.kx = 0.0
        self.ky = 0.0
        self.E = 0.0
        self.k_int = 0.4
        self.E_int = 0.1
        self.delay = 500
        self.delay_int = 1000
        self.norm = True
        self.subtract_neg = False

class MainWindow(QMainWindow):
    def __init__(self, filepath, title):
        super().__init__()
        self.setWindowTitle(title)
        self.ax_map = {}
        self.plots = {}
        self.state = State()
        self.dragging_crosshair = False
        self.drag_plot = None
        
        self.load_data(filepath)

        self.setup_ui()

        self.init_plots()     # create plot objects
        self.setup_plots()    # assign axes

    def load_data(self, filepath):

        try:
            loader = DataLoader(filepath)
            self.state.I = loader.load()

        except Exception as e:
            print(f"Failed to load data: {e}")
            self.state.I = None

    def setup_ui(self):

        # --------------------------------------------------
        # Central widget
        # --------------------------------------------------

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # --------------------------------------------------
        # Left control panel
        # --------------------------------------------------

        control_panel = QFrame()
        control_panel.setFixedWidth(250)

        control_layout = QVBoxLayout(control_panel)

        title = QLabel("Controls")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        control_layout.addWidget(title)

        # Energy slider

        control_layout.addWidget(QLabel("Energy (eV)"))

        self.energy_slider = QSlider(Qt.Horizontal)
        E_min = round(self.state.I.E.values.min(),1)
        E_max = round(self.state.I.E.values.max(),1)

        self.energy_slider.setMinimum(int(10*E_min))
        self.energy_slider.setMaximum(int(10*E_max))
        self.energy_slider.setValue(int(10*self.state.E))

        control_layout.addWidget(self.energy_slider)

        self.energy_label = QLabel(f"{self.state.E:.2f} eV")
        control_layout.addWidget(self.energy_label)

        # k integration

        control_layout.addWidget(QLabel("Δk (Å⁻¹)"))

        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setMinimum(1)
        self.k_slider.setMaximum(400)
        self.k_slider.setValue(int(self.state.k_int*100))

        control_layout.addWidget(self.k_slider)

        self.k_int_label = QLabel(f"{self.state.k_int:.2f} Å⁻¹")
        control_layout.addWidget(self.k_int_label)

        # Checkboxes

        self.edc_checkbox = QCheckBox("EDC")
        control_layout.addWidget(self.edc_checkbox)

        self.waterfall_checkbox = QCheckBox("Waterfall")
        control_layout.addWidget(self.waterfall_checkbox)

        self.kcut_checkbox = QCheckBox("Arbitrary k-cut")
        control_layout.addWidget(self.kcut_checkbox)
        
        # Button
        self.reset_button = QPushButton("Reset")
        control_layout.addWidget(self.reset_button)

        control_layout.addStretch()

        main_layout.addWidget(control_panel)

        # --------------------------------------------------
        # Matplotlib Figure
        # --------------------------------------------------

        self.fig = Figure(figsize=(10, 8))

        self.canvas = FigureCanvas(self.fig)

        self.axes = self.fig.subplots(2, 2).flatten()

        main_layout.addWidget(self.canvas)

        # Mouse Click and Movement
        self.fig.canvas.mpl_connect(
            "button_press_event",
            self.on_mouse_press
        )

        self.fig.canvas.mpl_connect(
            "motion_notify_event",
            self.on_mouse_move
        )

        self.fig.canvas.mpl_connect(
            "button_release_event",
            self.on_mouse_release
        )

        self.fig.canvas.mpl_connect("button_press_event", self.on_right_click)

        # --------------------------------------------------
        # Signals
        # --------------------------------------------------

        self.energy_slider.valueChanged.connect(
            self.update_energy
        )

        self.k_slider.valueChanged.connect(
            self.update_k_int
        )

    def setup_plots(self):
        self.ax_map["trace"] = self.axes[1]
        self.ax_map["mm"] = self.axes[0]

        self.plots["trace"].attach(self.ax_map["trace"])
        self.plots["mm"].attach(self.ax_map["mm"])

    def setup_plots(self):

        self.ax_map = {
            "mm": self.axes[0],
            "trace": self.axes[1]
        }

        for name, plot in self.plots.items():
            plot.attach(self.ax_map[name])

    def init_plots(self):

        self.plots = {
            "mm": MomentumMapPlot(self.axes[0], self.state),
            "trace": TimeTracePlot(self.axes[1], self.state)
        }

        self.refresh()
        self.canvas.draw()

    def create_plot(self, plot_type, ax):

        # clear axis first
        ax.clear()

        if plot_type == "trace":
            plot = TimeTracePlot(ax, self.state)
            plot.setup_axis()
            return plot

        elif plot_type == "edc":
            plot = EDCPlot(ax, self.state)
            plot.setup_axis()
            return plot
        
        elif plot_type == "mm":
            plot = MomentumMapPlot(ax, self.state)
            plot.setup_axis()
            return plot

        elif plot_type == "wf":
            plot = WaterfallPlot(ax, self.state)
            plot.setup_axis()
            return plot
        
        else:
            return None
        
    def set_plot(self, ax, plot_type):

        # 1. FULL reset (important)
        ax.clear()
        ax.set_title("")   # optional cleanup

        # 2. create new plot
        new_plot = self.create_plot(plot_type, ax)

        # 3. store
        self.plots[ax] = new_plot

        # 4. force clean redraw
        self.refresh()
    
    def refresh(self):
        for plot in self.plots.values():
            if plot:
                plot.update()

        self.canvas.draw_idle()

    def update_energy(self, value):

        self.state.E = value / 10
        self.energy_label.setText(f"{self.state.E:.2f} eV")
        self.refresh()

    def update_kx(self, value):

        self.state.kx = value

        self.refresh()

    def update_ky(self, value):

        self.state.ky = value

        self.refresh()

    def update_k_int(self, value):

        self.state.k_int = value / 100.0
        self.k_int_label.setText(fr"{self.state.k_int:.2f} Å⁻¹")

        self.refresh()

    def on_mouse_press(self, event):

        if event.button != 1:
            return

        if event.inaxes is None:
            return

        for plot in self.plots.values():

            if not isinstance(plot, MomentumMapPlot):
                continue

            if event.inaxes != plot.ax:
                continue

            self.dragging_crosshair = True
            self.drag_plot = plot

            self.state.kx = event.xdata
            self.state.ky = event.ydata

            self.refresh()

            break

    def on_mouse_move(self, event):

        if not self.dragging_crosshair:
            return

        if event.inaxes != self.drag_plot.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        self.state.kx = event.xdata
        self.state.ky = event.ydata

        self.refresh()

    def on_mouse_release(self, event):

        self.dragging_crosshair = False
        self.drag_plot = None

    def on_right_click(self, event):
        if event.button != 3:
            return

        ax = event.inaxes
        if ax is None:
            return

        self.show_context_menu(ax, event)

    def make_action(self, ax, label, plot_type):
        act = QAction(label, self)
        act.triggered.connect(lambda: self.set_plot(ax, plot_type))
        return act

    def show_context_menu(self, ax, event):
        menu = QMenu(self)

        actions = {
            "Dynamics (trace)": "trace",
            "EDC": "edc",
            "Momentum Map": "mm",
            "Waterfall": "wf"
        }

        for label, plot_type in actions.items():
            act = menu.addAction(label)
            act.triggered.connect(
                lambda checked=False, p=plot_type, a=ax: self.set_plot(a, p)
            )

        pos = None
        if event.guiEvent is not None:
            pos = event.guiEvent.globalPos()
        else:
            pos = self.mapToGlobal(self.cursor().pos())

        menu.exec_(pos)

def launch_gui():
    app = QApplication(sys.argv)

    filepath, _ = QFileDialog.getOpenFileName(
        None,
        "Select ARPES data file",
        "",
        "HDF5 (*.h5);;NetCDF (*.nc);;All files (*)"
    )

    if not filepath:
        print("No file selected.")
        return

    title = os.path.basename(filepath)

    window = MainWindow(filepath, title)
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    launch_gui()