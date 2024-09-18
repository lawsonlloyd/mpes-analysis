# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:34:02 2024

@author: lloyd
"""

#%%

def main():
    
    # Load data (I, ax_kx, ax_ky, ax_E_offset, ax_delay_offset)
    data_handler = DataHandler(I, ax_kx, ax_ky, ax_E_offset, ax_delay_offset)

    # Initialize plot manager
    plot_manager = PlotManager(data_handler)

    # Initialize sliders and attach update event
    slider_manager = SliderManager(plot_manager)
    slider_manager.E_slider.on_changed(slider_manager.on_slider_update)
    slider_manager.delay_slider.on_changed(slider_manager.on_slider_update)

    # Initialize event handler for interactivity
    event_handler = EventHandler(plot_manager)
    plot_manager.fig.canvas.mpl_connect('button_press_event', event_handler.on_press)
    plot_manager.fig.canvas.mpl_connect('button_release_event', event_handler.on_release)

    plt.show()

if __name__ == "__main__":
    main()