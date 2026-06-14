#launch_mpes_gui.py


def launch_gui():
    app = QApplication(sys.argv)

    filepath, _ = QFileDialog.getOpenFileName(
        None,
        "Select ARPES data file",
        "",
        "HDF5 files (*.h5);;NetCDF (*.nc);;All files (*)"
    )

    if not filepath:
        print("No file selected.")
        return

    title = os.path.basename(filepath)

    window = MainWindow(filepath, title)
    window.show()

    sys.exit(app.exec_())