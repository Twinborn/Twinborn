import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import time

class MangaBalloonDetector:
    """
    Backend logic for Manga Balloon Detection.
    Handles model loading and inference.
    """
    def __init__(self, model_path="best.pt"):
        self.model_path = model_path
        self.model = None
        self.device = 'cpu' # Default to cpu, can be 'cuda' or 'mps' if available

    def load_model(self):
        try:
            self.model = YOLO(self.model_path)
            # Warmup
            # self.model(np.zeros((640, 640, 3), dtype=np.uint8))
            return True, "Model loaded successfully."
        except Exception as e:
            return False, f"Error loading model: {e}"

    def predict(self, source, conf=0.25, iou=0.7, imgsz=640, save=False, save_txt=False, save_crop=False, project="runs/detect", name="exp"):
        """
        Runs inference on the source (path or image).
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        # Run inference
        results = self.model(
            source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=save,
            save_txt=save_txt,
            save_crop=save_crop,
            project=project,
            name=name,
            exist_ok=True
        )
        return results

class AdvancedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Manga Balloon Detector")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e2e")

        # Initialize Logic
        self.detector = MangaBalloonDetector(model_path=os.path.join(os.getcwd(), "best.pt"))

        # Styling
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TFrame", background="#1e1e2e")
        self.style.configure("TLabel", background="#1e1e2e", foreground="white")
        self.style.configure("TButton", background="#6c5ce7", foreground="white", font=("Arial", 10, "bold"))
        self.style.map("TButton", background=[("active", "#5b4bc4")])
        self.style.configure("TNotebook", background="#1e1e2e", borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#2d2d44", foreground="white", padding=[10, 5])
        self.style.map("TNotebook.Tab", background=[("selected", "#6c5ce7")])

        # Variables
        self.model_path_var = tk.StringVar(value=self.detector.model_path)
        self.conf_var = tk.DoubleVar(value=0.25)
        self.iou_var = tk.DoubleVar(value=0.7)
        self.imgsz_var = tk.IntVar(value=640)
        self.save_crop_var = tk.BooleanVar(value=False)
        self.save_txt_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()

        # Load model in background
        threading.Thread(target=self._initial_load_model, daemon=True).start()

    def _initial_load_model(self):
        self._update_status("Loading model...")
        success, msg = self.detector.load_model()
        if success:
            self._update_status(f"Ready. {msg}")
        else:
            self._update_status(f"Error: {msg}")
            messagebox.showerror("Model Error", msg)

    def _update_status(self, text):
        self.status_var.set(text)

    def _build_ui(self):
        # Main Layout: Sidebar (Settings) + Main Content
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Sidebar
        sidebar = ttk.Frame(main_container, width=250)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(sidebar, text="SETTINGS", font=("Arial", 14, "bold")).pack(pady=10, anchor="w")

        # Model Selection
        ttk.Label(sidebar, text="Model Path:").pack(anchor="w")
        ttk.Entry(sidebar, textvariable=self.model_path_var).pack(fill=tk.X, pady=5)
        ttk.Button(sidebar, text="Browse", command=self._browse_model).pack(fill=tk.X)
        ttk.Button(sidebar, text="Reload Model", command=self._reload_model).pack(fill=tk.X, pady=5)

        ttk.Separator(sidebar, orient='horizontal').pack(fill='x', pady=10)

        # Parameters
        ttk.Label(sidebar, text="Confidence Threshold:").pack(anchor="w")
        scale_conf = tk.Scale(sidebar, variable=self.conf_var, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, bg="#1e1e2e", fg="white", highlightthickness=0)
        scale_conf.pack(fill=tk.X)

        ttk.Label(sidebar, text="IoU Threshold (NMS):").pack(anchor="w")
        scale_iou = tk.Scale(sidebar, variable=self.iou_var, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, bg="#1e1e2e", fg="white", highlightthickness=0)
        scale_iou.pack(fill=tk.X)

        ttk.Label(sidebar, text="Image Size:").pack(anchor="w")
        ttk.Entry(sidebar, textvariable=self.imgsz_var).pack(fill=tk.X)

        ttk.Separator(sidebar, orient='horizontal').pack(fill='x', pady=10)

        # Outputs
        ttk.Checkbutton(sidebar, text="Save Crops", variable=self.save_crop_var, style="TCheckbutton").pack(anchor="w")
        ttk.Checkbutton(sidebar, text="Save Labels (TXT)", variable=self.save_txt_var, style="TCheckbutton").pack(anchor="w")

        # Content Area (Tabs)
        content_area = ttk.Frame(main_container)
        content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.notebook = ttk.Notebook(content_area)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Single Image
        self.tab_single = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_single, text="Single Image")
        self._build_single_image_tab()

        # Tab 2: Batch Processing
        self.tab_batch = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_batch, text="Batch Processing")
        self._build_batch_tab()

        # Status Bar
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#2d2d44", fg="#a0a0ff")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_single_image_tab(self):
        # Controls
        ctrl_frame = ttk.Frame(self.tab_single)
        ctrl_frame.pack(fill=tk.X, pady=10)

        ttk.Button(ctrl_frame, text="Select Image", command=self._select_image).pack(side=tk.LEFT, padx=5)

        # Display Area
        self.display_frame = ttk.Frame(self.tab_single)
        self.display_frame.pack(fill=tk.BOTH, expand=True)

        # Original
        self.panel_orig = tk.Label(self.display_frame, text="Original", bg="#2d2d44", fg="#cccccc")
        self.panel_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Result
        self.panel_res = tk.Label(self.display_frame, text="Result", bg="#2d2d44", fg="#cccccc")
        self.panel_res.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    def _build_batch_tab(self):
        ctrl_frame = ttk.Frame(self.tab_batch)
        ctrl_frame.pack(fill=tk.X, pady=10)

        ttk.Button(ctrl_frame, text="Select Source Folder", command=self._batch_select_folder).pack(side=tk.LEFT, padx=5)
        self.batch_folder_lbl = ttk.Label(ctrl_frame, text="No folder selected")
        self.batch_folder_lbl.pack(side=tk.LEFT, padx=10)

        ttk.Button(ctrl_frame, text="Start Batch Process", command=self._start_batch).pack(side=tk.RIGHT, padx=5)

        # Log area
        self.log_text = tk.Text(self.tab_batch, bg="#2d2d44", fg="white", font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pt")])
        if path:
            self.model_path_var.set(path)
            self._reload_model()

    def _reload_model(self):
        self.detector.model_path = self.model_path_var.get()
        threading.Thread(target=self._initial_load_model, daemon=True).start()

    def _select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if path:
            self._process_single_image(path)

    def _process_single_image(self, path):
        self._update_status("Processing image...")

        def run():
            try:
                results = self.detector.predict(
                    path,
                    conf=self.conf_var.get(),
                    iou=self.iou_var.get(),
                    imgsz=self.imgsz_var.get(),
                    save=True,
                    save_crop=self.save_crop_var.get(),
                    save_txt=self.save_txt_var.get()
                )[0]

                # Update UI
                self.root.after(0, lambda: self._show_results(path, results))
            except Exception as e:
                self.root.after(0, lambda: self._update_status(f"Error: {e}"))

        threading.Thread(target=run, daemon=True).start()

    def _show_results(self, orig_path, result):
        try:
            # Result path
            save_dir = result.save_dir
            filename = os.path.basename(orig_path)
            res_path = os.path.join(save_dir, filename)

            # Helper to resize keeping aspect ratio
            def resize_contain(img, size):
                img.thumbnail(size, Image.LANCZOS)
                return img

            # Load images
            img_orig = Image.open(orig_path)
            img_res = Image.open(res_path)

            # Resize for display (fixed height 500)
            target_h = 500
            aspect = img_orig.width / img_orig.height
            target_w = int(target_h * aspect)

            img_orig = resize_contain(img_orig, (target_w, target_h))
            img_res = resize_contain(img_res, (target_w, target_h))

            tk_orig = ImageTk.PhotoImage(img_orig)
            tk_res = ImageTk.PhotoImage(img_res)

            self.panel_orig.config(image=tk_orig, text="")
            self.panel_orig.image = tk_orig
            self.panel_res.config(image=tk_res, text="")
            self.panel_res.image = tk_res

            count = len(result.boxes)
            self._update_status(f"Processed {filename}. Detected {count} objects.")

        except Exception as e:
            self._update_status(f"Error displaying results: {e}")

    def _batch_select_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.batch_folder_path = path
            self.batch_folder_lbl.config(text=path)

    def _start_batch(self):
        if not hasattr(self, 'batch_folder_path'):
            messagebox.showwarning("Warning", "Please select a folder first.")
            return

        self._update_status("Batch processing started...")
        self.log_text.delete(1.0, tk.END)

        def run_batch():
            files = [f for f in os.listdir(self.batch_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            total = len(files)

            for i, f in enumerate(files):
                filepath = os.path.join(self.batch_folder_path, f)
                try:
                    self.detector.predict(
                        filepath,
                        conf=self.conf_var.get(),
                        iou=self.iou_var.get(),
                        imgsz=self.imgsz_var.get(),
                        save=True,
                        save_crop=self.save_crop_var.get(),
                        save_txt=self.save_txt_var.get(),
                        name="batch_run"
                    )
                    msg = f"[{i+1}/{total}] Processed {f}\n"
                except Exception as e:
                    msg = f"[{i+1}/{total}] Error processing {f}: {e}\n"

                self.root.after(0, lambda m=msg: self.log_text.insert(tk.END, m))
                self.root.after(0, lambda: self.log_text.see(tk.END))

            self.root.after(0, lambda: self._update_status(f"Batch processing complete. {total} files."))

        threading.Thread(target=run_batch, daemon=True).start()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = AdvancedApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        print("This application requires a display environment (X11/Wayland) to run the GUI.")
