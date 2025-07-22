# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:01:16 2024

@author: lloyd
"""
#Main.py
import matplotlib.pyplot as plt

from Manager import DataHandler, FigureHandler, PlotHandler, ValueHandler, SliderManager, EventHandler, CheckButtonManager, ClickButtonManager, ArbitraryCutHandler

def main(I):
    
    value_manager =  ValueHandler()
    data_handler = DataHandler(value_manager, I)

    # Initialize plot manager and check and click button managers
    figure_handler = FigureHandler()
    check_button_manager = CheckButtonManager()
    plot_manager = PlotHandler(figure_handler, data_handler, value_manager, check_button_manager)
    click_button_manager = ClickButtonManager(plot_manager, check_button_manager)
    arb_cut_handler = ArbitraryCutHandler(plot_manager, data_handler)

    # Initialize sliders and attach update event
    slider_manager = SliderManager(value_manager, plot_manager, check_button_manager)
    slider_manager.E_slider.on_changed(slider_manager.on_slider_update)
    slider_manager.E_int_slider.on_changed(slider_manager.on_slider_update)
    slider_manager.k_int_slider.on_changed(slider_manager.on_slider_update)
    slider_manager.delay_slider.on_changed(slider_manager.on_slider_update)
    slider_manager.delay_int_slider.on_changed(slider_manager.on_slider_update)

    # Initialize event handler for interactivity
    event_handler = EventHandler(value_manager, slider_manager, plot_manager, check_button_manager, arb_cut_handler)
    plot_manager.fig.canvas.mpl_connect('button_press_event', event_handler.on_press)
    plot_manager.fig.canvas.mpl_connect('motion_notify_event', event_handler.on_motion)
    plot_manager.fig.canvas.mpl_connect('button_release_event', event_handler.on_release)

    plt.show(block=True)


if __name__ == "__main__":
    
    main(I)
