import tkinter as tk
from tkinter import filedialog, ttk
import multiprocessing
import os

from Main import main

def show_splash_screen():
    splash = tk.Toplevel()
    splash.overrideredirect(True)
    splash.geometry("300x150+600+300")
    splash.configure(bg="white")

    label = ttk.Label(
        splash,
        text="Launching ARPES GUI...",
        font=("Helvetica", 14)
    )
    label.place(relx=0.5, rely=0.5, anchor="center")

    splash.update()
    return splash


def launch_gui(filepath):
    title = os.path.basename(filepath)

    p = multiprocessing.Process(
        target=main,
        args=(filepath, title),   # ✅ pass filepath only
        daemon=False
    )
    p.start()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    root = tk.Tk()
    root.withdraw()  # single hidden root

    filepath = filedialog.askopenfilename(
        title="Select ARPES data file",
        filetypes=[("All files", "*.*")]
    )

    if filepath:
        splash = show_splash_screen()
        root.update()

        launch_gui(filepath)

        splash.destroy()  # ✅ close splash once launched

    root.destroy()
