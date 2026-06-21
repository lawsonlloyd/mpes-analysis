from arpes_tools import mpes
from arpes_tools.mpes import cmap_LTL, cmap_LTL2
from arpes_tools.loader import DataLoader
#from gui.fake_data import make_fake_trarpes_data

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
    QFileDialog,
    QDoubleSpinBox
)

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor

#from superqt import QRangeSlider
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class State:
    def __init__(self):
        self.I = None #make_fake_trarpes_data()
        self.kx = 0.0
        self.ky = 0.0
        self.E = 0.0
        self.k_int = 0.4
        self.E_int = 0.1
        self.delay = 500
        self.delay_int = 1000
        self.E_enhance = None
        self.E_view_min = -2
        self.E_view_max = 3
        self.norm = True
        self.subtract_neg = False
        self.cmap = cmap_LTL

class BasePlot:
    def __init__(self, ax, state):
        self.ax = ax
        self.state = state

    dependencies = set()
    crosshair_dependencies = set()

    def setup_axis(self):
        pass
    
    def context_actions(self):
        return {}
    #def attach(self, ax):
    #    self.ax = ax

    def apply_energy_limits(self):
        pass

    def update_crosshairs(self):
        pass

    def update(self):
        raise NotImplementedError

class MomentumMapPlot(BasePlot):

    def __init__(self, ax, state):
        super().__init__(ax, state)
        _, _, self.im, = mpes.plot_momentum_maps(
            self.state.I,
            self.state.E, self.state.E_int,
            self.state.delay, self.state.delay_int,
            fig = self.ax.figure, ax=ax, cmap = cmap_LTL
        )

        self.dependencies = {
            "E",
            "E_int",
            "delay",
            "delay_int"
        }

        self.crosshair_dependencies = {
            "kx",
            "ky",
            "k_int",
        }

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

    def update_crosshairs(self):
        self.vline.set_xdata([self.state.kx, self.state.kx])
        self.hline.set_ydata([self.state.ky, self.state.ky])

    def update(self):
        frame = mpes.get_momentum_map(
            self.state.I,
            self.state.E, self.state.E_int,
            self.state.delay, self.state.delay_int,
            norm = self.state.norm,
            subtract_neg = self.state.subtract_neg
        )

        if self.state.subtract_neg:
            self.im.set_cmap("seismic")
            vmax = np.nanmax(np.abs(frame))
            self.im.set_clim(-vmax, vmax)
        else:
            self.im.set_cmap(self.state.cmap)
            self.im.set_clim(0, 1)

        self.im.set_data(frame)
        self.ax.set_title(f"{self.state.E:.2f} eV")

class EDCPlot(BasePlot):

    def context_actions(self):
        return {
            "Fit Peak": self.fit_peak
        }
    
    def fit_peak(self):

        #result = fit_gaussian
        print('Fitting the EDC!')

    def __init__(self, ax, state):
        super().__init__(ax, state)   

        self.dependencies = {
            "kx",
            "ky",
            "k_int",
            "norm",
            "subtract_neg",
            "E_enhance"
        }

        _, _, self.line, self.line_enhance = mpes.plot_edc(
            self.state.I,
            (self.state.kx, self.state.ky),
            (self.state.k_int, self.state.k_int),
            delay = self.state.delay, delay_int = self.state.delay_int,
            norm_trace = self.state.norm,
            subtract_neg = self.state.subtract_neg,
            E_enhance = self.state.E_enhance,
            fig = self.ax.figure, ax=ax
        )
    
    def apply_energy_limits(self):
        self.ax.set_xlim(
            self.state.E_view_min,
            self.state.E_view_max
        )

    def setup_axis(self):
        self.ax.set_aspect("auto")
        self.ax.set_ylim(0, 1)

    def update_crosshairs(self):
            self.line_enhance.set_xdata([self.E_enhance])

    def update(self):
        edc = mpes.get_edc(
            self.state.I,
            (self.state.kx, self.state.ky),
            (self.state.k_int, self.state.k_int),
            delay = self.state.delay, delay_int = self.state.delay_int,
            norm_trace = self.state.norm,
            subtract_neg = self.state.subtract_neg,
            E_enhance = self.state.E_enhance,
        )

        if self.state.subtract_neg is True:
            self.ax.set_ylim(-1.05*np.abs(edc.min()),1.05*edc.max())
        else:
            self.ax.set_ylim(0,1.05)

        self.line[0].set_ydata(edc)
        #self.line.set_data(self.state.I.E.values, edc)

class TimeTracePlot(BasePlot):

    def context_actions(self):
        return {
            "Fit Exponential": self.fit_exponential
        }
    
    def fit_exponential(self):
        
        trace = mpes.get_time_trace(
            self.state.I,
            self.state.E,
            self.state.E_int,
            (self.state.kx, self.state.ky),
            (self.state.k_int, self.state.k_int),
            norm_trace=self.state.norm,
            subtract_neg=self.state.subtract_neg,
        )

        delays = self.state.I.delay.values
        #result = fit_function(delays, trace)

        print('haha, do the fit!')

    def __init__(self, ax, state):
        super().__init__(ax, state)

        self.dependencies = {
            "kx",
            "ky",
            "k_int",
            "E",
            "E_int",
            "norm",
            "subtract_neg"
        }
        _, _, self.trace, = mpes.plot_time_traces(
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
        self.ax.set_ylim(0,1.05)

    def update(self):
        self.time_trace = mpes.get_time_trace(
            self.state.I,
            self.state.E, self.state.E_int,
            (self.state.kx, self.state.ky),
            (self.state.k_int, self.state.k_int),
            norm_trace = self.state.norm,
            subtract_neg = self.state.subtract_neg
        )

        #trace = trace / trace.max()
        self.trace.set_xdata(self.state.I.delay.values)
        self.trace.set_ydata(self.time_trace)
        
        if self.state.subtract_neg is True:
            self.ax.set_ylim(-1.05*np.abs(self.time_trace.min()),1.05*self.time_trace.max())
        else:
            self.ax.set_ylim(0,1.05)

class kEPlot(BasePlot):

    def __init__(self, ax, state):
        super().__init__(ax, state)
        _, _, self.im, = mpes.plot_kx_frame(
            self.state.I,
            self.state.ky, self.state.k_int,
            self.state.delay, self.state.delay_int,
            subtract_neg = self.state.subtract_neg,
            E_enhance = self.state.E_enhance,
            energy_limits = [-3,3],
            fig = self.ax.figure, ax=ax, cmap = cmap_LTL
        )

        self.hline = self.ax.axhline(
            -10,
            color="black",
            ls="--"
        )

        self.dependencies = {
            "ky",
            "k_int",
            "delay",
            "delay_int",
            "norm",
            "subtract_neg",
            "E_enhance"
        }

        self.crosshair_dependencies = {
            "E_enhance"
        }

    def apply_energy_limits(self):
        self.ax.set_ylim(
            self.state.E_view_min,
            self.state.E_view_max
        )

    def setup_axis(self):
        self.ax.set_aspect("auto")
        self.im.set_clim(0, 1)

    def update_crosshairs(self):
            self.hline.set_ydata([self.E_enhance, self.E_enhance])

    def update(self):
        frame = mpes.get_kx_E_frame(
            self.state.I,
            self.state.ky, self.state.k_int,
            self.state.delay, self.state.delay_int,
            subtract_neg = self.state.subtract_neg,
            E_enhance = self.state.E_enhance
        )

        if self.state.subtract_neg:
            self.im.set_cmap("seismic")
            vmax = np.max(np.abs(frame))
            self.im.set_clim(-vmax, vmax)
        else:
            self.im.set_cmap(self.state.cmap)
            self.im.set_clim(0, 1)

        self.im.set_data(frame.T)
        self.ax.set_title(f"{self.state.ky:.1f} Å⁻¹")

class WaterfallPlot(BasePlot):

    def __init__(self, ax, state):
        super().__init__(ax, state)
        _, _, self.im, = mpes.plot_waterfall(
            self.state.I,
            self.state.kx, self.state.k_int,
            self.state.ky, self.state.k_int,
            subtract_neg = self.state.subtract_neg,
            E_enhance = self.state.E_enhance,
            energy_limits = [-1,3],
            fig = self.ax.figure, ax=ax, cmap = cmap_LTL
        )
        self.hline = self.ax.axhline(
            -10,
            color="black",
            ls="--"
        )

        self.crosshair_dependencies = {
            "E_enhance"
        }

        self.dependencies = {
            "kx",
            "ky",
            "k_int",
            "norm",
            "subtract_neg",
            "E_enhance"
        }

    def apply_energy_limits(self):
        self.ax.set_ylim(
            self.state.E_view_min,
            self.state.E_view_max
        )

    def setup_axis(self):
        self.ax.set_aspect("auto")
        #self.ax.set_ylim(-1, 3)
        self.im.set_clim(0, 1)

    def update_crosshairs(self):
        self.hline.set_ydata([self.state.E_enhance, self.state.E_enhance])

    def update(self):
        frame = mpes.get_waterfall(
            self.state.I,
            self.state.kx, self.state.k_int,
            self.state.ky, self.state.k_int,
            subtract_neg = self.state.subtract_neg,
            E_enhance = self.state.E_enhance,
        )
        #frame = frame / frame.max()

        self.im.set_data(frame)

        if self.state.subtract_neg:
            self.im.set_cmap("seismic")
            vmax = np.max(np.abs(frame))
            self.im.set_clim(-vmax, vmax)
        else:
            self.im.set_cmap(self.state.cmap)
            self.im.set_clim(0, 1)

class MainWindow(QMainWindow):
    def __init__(self, filepath, title):
        super().__init__()
        self.setWindowTitle(title)
        #self.ax_map = {}
        self.state = State()
        self.load_data(filepath)
        self.plots = {}
        self.dragging_crosshair = False
        self.drag_plot = None
        
        self.setup_ui()
        self.init_plots()     # create plot objects
        #self.setup_plots()    # assign axes

    def load_data(self, filepath):

        try:
            loader = DataLoader(filepath)
            self.state.I = loader.load()
            self.state.I = self.state.I / self.state.I.max()
            
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

        # Delay Integration

        control_layout.addWidget(QLabel("E Integration (meV)"))

        self.energy_int_box = QDoubleSpinBox()
        self.energy_int_box.setRange(0, 2000)
        self.energy_int_box.setSingleStep(20)
        self.energy_int_box.setValue(100)
        control_layout.addWidget(self.energy_int_box)

        # k integration

        control_layout.addWidget(QLabel("Δk (Å⁻¹)"))

        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setMinimum(1)
        self.k_slider.setMaximum(400)
        self.k_slider.setValue(int(self.state.k_int*100))

        control_layout.addWidget(self.k_slider)

        self.k_int_label = QLabel(f"{self.state.k_int:.2f} Å⁻¹")
        control_layout.addWidget(self.k_int_label)

        # Delay Time

        control_layout.addWidget(QLabel("Delay (fs)"))

        self.delay_slider = QSlider(Qt.Horizontal)
        self.delay_slider.setMinimum(int(self.state.I.delay.values.min()))
        self.delay_slider.setMaximum(int(self.state.I.delay.values.max()))
        self.delay_slider.setValue(int(self.state.delay))

        control_layout.addWidget(self.delay_slider)

        self.delay_label = QLabel(f"{self.state.delay} fs")
        control_layout.addWidget(self.delay_label)

        # Delay Integration

        control_layout.addWidget(QLabel("Delay Integration (fs)"))

        self.delay_int_box = QDoubleSpinBox()
        self.delay_int_box.setRange(0, 1000)
        self.delay_int_box.setSingleStep(20)
        self.delay_int_box.setValue(200)
        control_layout.addWidget(self.delay_int_box)

        # Energy View Range

        # self.energy_range = QRangeSlider(Qt.Horizontal)
        # self.energy_range.setRange(
        #     int(E_min*10),
        #     int(E_max*10)
        # )
        # self.energy_range.setValue((-30,10))

        # Enhance Signal

        control_layout.addWidget(QLabel("Enhance Above:"))

        self.enhance_box = QDoubleSpinBox()
        self.enhance_box.setRange(-1, 5.0)
        self.enhance_box.setSingleStep(0.1)
        self.enhance_box.setValue(0)
        control_layout.addWidget(self.enhance_box)

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

        self.fig.canvas.mpl_connect("button_release_event", self.on_right_click)

        # --------------------------------------------------
        # Signals
        # --------------------------------------------------

        self.energy_slider.valueChanged.connect(
            self.update_energy
        )

        self.energy_int_box.valueChanged.connect(
            self.update_energy_int
        )

        self.k_slider.valueChanged.connect(
            self.update_k_int
        )

        self.delay_slider.valueChanged.connect(
            self.update_delay
        )

        self.delay_int_box.valueChanged.connect(
            self.update_delay_int
        )

        self.enhance_box.valueChanged.connect(
            self.update_E_enhance
        )

    # def setup_plots(self):

    #     self.ax_map = {
    #         "mm": self.axes[0],
    #         "trace": self.axes[1]
    #     }

    #     for name, plot in self.plots.items():
    #         plot.attach(self.ax_map[name])

    def init_plots(self):

        self.plots = {
            self.axes[0]: MomentumMapPlot(self.axes[0], self.state),
            self.axes[1]: TimeTracePlot(self.axes[1], self.state)
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

        elif plot_type == "kx":
            plot = kEPlot(ax, self.state)
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
    
    # def refresh(self):
    #     for plot in self.plots.values():
    #         if plot:
    #             plot.update()
    #             #plot.setup_axis()
    #             plot.apply_energy_limits()

    #     self.canvas.draw_idle()

    def refresh(self, changed_vars=None):

        for plot in self.plots.values():

            if changed_vars is None:
                plot.update()

            elif plot.dependencies & changed_vars:
                plot.update()

            #plot.apply_energy_limits()

        self.canvas.draw_idle()

    def refresh_crosshairs(self, changed_vars=None):

        for plot in self.plots.values():

            if changed_vars is None:
                plot.update_crosshairs()

            elif plot.crosshair_dependencies & changed_vars:
                plot.update_crosshairs()

            #plot.apply_energy_limits()

        self.canvas.draw_idle()

    def update_energy(self, value):

        self.state.E = value / 10
        self.energy_label.setText(f"{self.state.E:.2f} eV")
        self.refresh({"E"})

    def update_energy_int(self, value):
        self.state.E_int = value / 1000
        self.refresh({"E_int"})

    def update_kx(self, value):

        self.state.kx = value

        self.refresh({"kx"})
        self.refresh_crosshairs({"kx"})

    def update_ky(self, value):

        self.state.ky = value

        self.refresh({"ky"})
        self.refresh_crosshairs({"ky"})

    def update_k_int(self, value):

        self.state.k_int = value / 100.0
        self.k_int_label.setText(fr"{self.state.k_int:.2f} Å⁻¹")

        self.refresh({"k_int"})

    def update_delay(self, value):

        self.state.delay = value
        self.delay_label.setText(fr"{self.state.delay} fs")

        self.refresh({"delay"})

    def update_delay_int(self, value):
        self.state.delay_int = value
        self.refresh({"delay_int"})

    def update_E_enhance(self, value):
        self.state.E_enhance = value
        self.refresh({"E_enhance"})

    def toggle_subtract_neg(self, checked):

        self.state.subtract_neg = checked

        self.refresh({"subtract_neg"})

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

            self.refresh({"kx", "ky"})
            self.refresh_crosshairs({"kx", "ky"})

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

        self.refresh({"kx", "ky"})
        self.refresh_crosshairs({"kx", "ky"})

    def on_mouse_release(self, event):

        self.dragging_crosshair = False
        self.drag_plot = None

    def on_right_click(self, event):
        if event.button != 3:
            return

        ax = event.inaxes
        if ax is None:
            return
    
        print("RIGHT CLICK")
        self.show_context_menu(ax, event)

    def make_action(self, ax, label, plot_type):
        act = QAction(label, self)
        act.triggered.connect(lambda: self.set_plot(ax, plot_type))
        return act

    def add_check_action(self, menu, label, checked, callback):
        act = QAction(label, self)
        act.setCheckable(True)
        act.setChecked(checked)
        act.triggered.connect(callback)
        menu.addAction(act)
        return act

    def show_context_menu(self, ax, event):
        menu = QMenu(self)

        panel_menu = menu.addMenu("Change Panel")
        processing_menu = menu.addMenu("Processing")
        #analysis_menu = menu.addMenu("Analysis")

        actions = {
            "Dynamics": "trace",
            "EDC": "edc",
            "Momentum Map": "mm",
            "Waterfall": "wf",
            "E-k": "kx",
        }

        for label, plot_type in actions.items():
            act = panel_menu.addAction(label)
            act.triggered.connect(
                lambda checked=False, p=plot_type, a=ax:
                self.set_plot(a, p)
            )

        self.add_check_action(
            processing_menu,
            "Subtract Negative Delays",
            self.state.subtract_neg,
            self.toggle_subtract_neg
        )

        print("MENU ABOUT TO OPEN")

        ###
        # pos = None
        # if event.guiEvent is not None:
        #     pos = event.guiEvent.globalPos()
        # else:
        #     pos = self.mapToGlobal(self.cursor().pos())

        # menu.exec_(pos)
        menu.exec_(QCursor.pos())

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