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

    data_loader = DataLoader(filepath)
    I = data_loader.load()

    title = os.path.basename(filepath)  # ✅ Only the filename, not full path

    #I = load_from_file(filepath)

    # Launch the GUI in a subprocess to allow multiple windows
    multiprocessing.set_start_method("spawn", force=True)
    p = multiprocessing.Process(target=main, args=(I, title))
    p.start()

if __name__ == "__main__":
    # Hide root tkinter window
    root = tk.Tk()
    Tk().withdraw()  # Hide base window

    # Prompt user for file selection
    filepath = filedialog.askopenfilename(
        title="Select ARPES data file",
        filetypes=[("All files", "*.*")]
    )
    
    root.destroy()  # Clean up

    if filepath:
        launch_gui(filepath)
        