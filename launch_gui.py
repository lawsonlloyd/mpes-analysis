import tkinter as tk
from tkinter import filedialog
from tkinter import Tk
import multiprocessing
import os
from Main import main
from Loader import DataLoader

def show_splash_screen():
    splash = tk.Tk()
    splash.overrideredirect(True)
    splash.geometry("300x150+600+300")  # size + position (tweak as needed)
    splash.configure(bg="white")

    label = ttk.Label(splash, text="Launching ARPES GUI...", font=("Helvetica", 14))
    label.place(relx=0.5, rely=0.5, anchor="center")

    splash.update()
    return splash

def launch_gui(filepath):
    splash = show_splash_screen()

    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select ARPES data file",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    root.destroy()

    if not filepath:
        splash.destroy()
        print("No file selected.")
        return
    
    title = os.path.basename(filepath)

    try:
        data_loader = DataLoader(filepath)
        I = data_loader.load()
    except Exception as e:
        splash.destroy()
        print(f"Error loading data: {e}")
        return

    splash.destroy()

    multiprocessing.set_start_method("spawn", force=True)
    multiprocessing.Process(target=main, args=(I, title)).start()

if __name__ == "__main__":
    launch_gui(filepath)
        