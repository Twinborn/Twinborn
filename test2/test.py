# manga_balon_bulucu_kolay.py  ← bu dosyayı oluştur

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO
from PIL import Image, ImageTk
import os

# MODEL YOLU
MODEL_PATH = "best.pt"  # best.pt aynı klasörde olsun

# Modeli yükle
try:
    model = YOLO(MODEL_PATH)
    print("Model yüklendi!")
except Exception as e:
    messagebox.showerror("Hata", f"best.pt bulunamadı veya bozuk!\nHata: {e}")
    exit()

# Ana pencere
root = tk.Tk()
root.title("Manga Balon Bulucu - Senin Modelin")
root.geometry("1100x750")
root.configure(bg="#1e1e2e")

# Başlık
tk.Label(root, text="Manga Balon Tespit Robotu", font=("Arial", 24, "bold"), fg="#ffffff", bg="#1e1e2e").pack(pady=20)

# Resim alanları
frame = tk.Frame(root, bg="#1e1e2e")
frame.pack(pady=10)

img_label = tk.Label(frame, text="Orijinal resim burada görünecek", bg="#2d2d44", fg="#cccccc", width=60, height=25)
img_label.pack(side=tk.LEFT, padx=20)

result_label = tk.Label(frame, text="Balonlu sonuç burada görünecek", bg="#2d2d44", fg="#cccccc", width=60, height=25)
result_label.pack(side=tk.RIGHT, padx=20)

# Durum
status = tk.Label(root, text="Hazır - Resim seçmek için butona tıkla", fg="#a0a0ff", bg="#1e1e2e", font=("Arial", 11))
status.pack(pady=10)

def process_image(path):
    status.config(text="İşleniyor... Lütfen bekle")
    root.update()

    try:
        results = model(path, save=True, exist_ok=True)[0]
        result_path = results.save_dir + "/" + os.path.basename(path)

        # Orijinal
        img = Image.open(path).resize((450, 600))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk, text="")
        img_label.image = img_tk

        # Sonuç
        res_img = Image.open(result_path).resize((450, 600))
        res_tk = ImageTk.PhotoImage(res_img)
        result_label.config(image=res_tk, text="")
        result_label.image = res_tk

        status.config(text=f"Başarılı! → {os.path.basename(path)}")
    except Exception as e:
        status.config(text=f"Hata: {e}")

def select_file():
    path = filedialog.askopenfilename(
        title="Manga sayfası seç",
        filetypes=[("Resim", "*.jpg *.jpeg *.png *.webp")]
    )
    if path:
        process_image(path)

# Büyük buton
btn = tk.Button(root, text="MANGA SAYFASI SEÇ", command=select_file,
                font=("Arial", 16, "bold"), bg="#6c5ce7", fg="white", height=3, width=30)
btn.pack(pady=30)

# Alt bilgi
tk.Label(root, text="Model: YOLOv8n - 720 resimle eğitildi - %95 doğruluk", fg="#74b9ff", bg="#1e1e2e").pack(side=tk.BOTTOM, pady=20)

root.mainloop()