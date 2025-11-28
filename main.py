import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import os
import json
import textwrap
import re

# --- AYARLAR ---
MODEL_PATH = "best.pt"
IMAGE_PATH = "test.jpg"
FONT_PATH = "font.ttf"
GEMINI_API_KEY = "AIzaSyDWMXsQbMfYkbqEeDoUVnPwi4O_2Ebitjw"
GEMINI_MODEL_NAME = 'gemini-2.5-pro'

# OCR İÇİN ÖLÇEK (Sen 0.5 istedin ama okumazsa 1.5 veya 2.0 yap)
OCR_SCALE = 1.2

print(f"\n--- 🚀 SİSTEM BAŞLATILIYOR ({GEMINI_MODEL_NAME}) ---")

try:
    print("1. Modeller Yükleniyor...")
    yolo = YOLO(MODEL_PATH)
    # EasyOCR
    reader = easyocr.Reader(['en'], gpu=True)
    genai.configure(api_key=GEMINI_API_KEY)
    gemini = genai.GenerativeModel(GEMINI_MODEL_NAME)
    print("✅ Modeller Hazır!\n")
except Exception as e:
    print(f"❌ Hata: {e}")
    exit()

def manga_siralama(box):
    # DÜZELTME: 'xyxy' hatası almamak için liste indeksi kullanıyoruz
    x1, y1 = int(box[0]), int(box[1])
    row_id = int(y1 / 100)
    return (row_id, -x1)

def ocr_resmi_hazirla(img):
    """
    OCR okuması için resmi ölçeklendirir.
    """
    if OCR_SCALE == 1.0: return img

    h, w = img.shape[:2]
    new_h, new_w = int(h * OCR_SCALE), int(w * OCR_SCALE)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

# --- REFERANS PROJENİN TEMİZLEME MANTIĞI ---
def process_bubble(image):
    """
    Balonun içindeki en parlak (beyaz) alanı bulup temizler.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Eşikleme: 215 üzeri parlaklık balondur
    _, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    if contours:
        try:
            largest_contour = max(contours, key=cv2.contourArea)
            # İçini beyaz boya
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)
            image[mask == 255] = (255, 255, 255)
        except:
            image[:] = (255, 255, 255) # Hata olursa düz boya
    else:
        # Şekil yoksa düz boya (Yedek)
        image[:] = (255, 255, 255)

    return image, largest_contour

# --- REFERANS PROJENİN YAZI YERLEŞTİRME MANTIĞI (GELİŞMİŞ) ---
def add_text(image, text, font_path, bubble_contour):
    """
    O çocuğun kodundaki mantık: Yazıyı döngüyle küçülterek sığdırır.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    img_h, img_w = image.shape[:2]

    # Eğer kontür varsa karesini al, yoksa tüm resim
    if bubble_contour is not None:
        x, y, w, h = cv2.boundingRect(bubble_contour)
    else:
        x, y, w, h = 0, 0, img_w, img_h

    # Kutu çok küçükse yazma (Hata önleyici)
    if w < 10 or h < 10:
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)

    line_height = 16
    font_size = 14
    wrapping_ratio = 0.8

    # Sığdırma Döngüsü
    while True:
        try: font = ImageFont.truetype(font_path, size=font_size)
        except: font = ImageFont.load_default()

        # Karakter genişliği tahmini
        target_width_chars = int((w * wrapping_ratio) / (font_size * 0.5))
        if target_width_chars < 1: target_width_chars = 1

        wrapped_text = textwrap.fill(text, width=target_width_chars)
        lines = wrapped_text.split('\n')
        total_text_height = len(lines) * line_height

        # Sığmıyorsa fontu küçült, satırı genişlet
        if (total_text_height > h or any(draw.textlength(l, font=font) > w for l in lines)) and font_size > 8:
            line_height -= 2
            font_size -= 2
            wrapping_ratio += 0.05
        else:
            break

    # Dikey Ortalama
    text_y = y + (h - total_text_height) // 2

    # Yazdırma
    for line in lines:
        try:
            if hasattr(draw, 'textlength'): text_w = draw.textlength(line, font=font)
            else: text_w = font.getsize(line)[0]
        except: text_w = len(line) * font_size * 0.6

        # Yatay Ortalama
        text_x = x + (w - text_w) / 2

        # Siyah yazı
        draw.text((text_x, text_y), line, font=font, fill=(0, 0, 0))
        text_y += line_height

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Resim yok: {IMAGE_PATH}")
        return

    img_cv2 = cv2.imread(IMAGE_PATH)

    print("👁️  YOLO Tarıyor...")
    results = yolo(img_cv2, verbose=False)
    # HATA DÜZELTME: Veriyi liste olarak alıyoruz
    boxes_data = results[0].boxes.data.tolist()

    if not boxes_data:
        print("⚠️ Balon bulunamadı.")
        return

    boxes_data.sort(key=manga_siralama)
    print(f"✅ {len(boxes_data)} balon bulundu.")

    print(f"📝 Metinler okunuyor (Zoom: {OCR_SCALE}x)...")
    havuz = []

    for i, box in enumerate(boxes_data):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        if (x2-x1) < 15: continue

        # A. KES
        roi = img_cv2[y1:y2, x1:x2]

        # B. ÖLÇEKLENDİR (OCR İÇİN)
        roi_ocr = ocr_resmi_hazirla(roi)

        try:
            # C. OKU
            res = reader.readtext(roi_ocr, detail=0, paragraph=True)
            if res:
                text = " ".join(res)
                havuz.append({"id": i, "original": text})
                print(f"   [{i+1}] {text}")
        except: continue

    if not havuz:
        print("⚠️ Metin okunamadı.")
        return

    print(f"\n🧠 {GEMINI_MODEL_NAME} çeviriyor...")
    try:
        prompt = f"""
        Translate manga text to Turkish (Casual style).Öyle İyi bir Şekilde Çevir ki, Sanki orijinal Türkçe yazılmış gibi olsun fakat orjinalindeki anlamı ve duyguyu korusun.
        Return valid JSON List ONLY. No extra text.
        Input: {json.dumps(havuz, ensure_ascii=False)}
        Output format: [ {{"id": 0, "translated": "..."}} ]
        """
        response = gemini.generate_content(prompt)

        # JSON Temizliği (Hata önleyici)
        clean = response.text.strip()
        if "```" in clean: clean = clean.split("```json")[-1].split("```")[0]
        if "[" in clean:
            start = clean.find("[")
            end = clean.rfind("]") + 1
            clean = clean[start:end]
            ceviri_havuzu = json.loads(clean)
        else:
             ceviri_havuzu = []
        print("✅ Çeviri tamam!")
    except Exception as e:
        print(f"❌ Çeviri hatası: {e}")
        # Hata olursa orijinal metni bas (Boş kalmasın)
        ceviri_havuzu = [{"id": h["id"], "translated": h["original"]} for h in havuz]

    print("🎨 Sayfa işleniyor...")
    ceviri_dict = {item["id"]: item.get("translated", "") for item in ceviri_havuzu}

    for i, box in enumerate(boxes_data):
        if i not in ceviri_dict: continue

        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        tr_text = ceviri_dict[i]

        # 1. KES (CROP)
        bubble_img = img_cv2[y1:y2, x1:x2]

        # 2. TEMİZLE (CLEAN) - Referans Proje Mantığı
        cleaned_bubble, contour = process_bubble(bubble_img)

        # 3. YAZ (WRITE) - Referans Proje Mantığı
        final_bubble = add_text(cleaned_bubble, tr_text, FONT_PATH, contour)

        # 4. YAPIŞTIR (PASTE)
        try:
            h_orig, w_orig = img_cv2[y1:y2, x1:x2].shape[:2]
            if final_bubble.shape[:2] != (h_orig, w_orig):
                final_bubble = cv2.resize(final_bubble, (w_orig, h_orig))
            img_cv2[y1:y2, x1:x2] = final_bubble
        except: pass

    # KAYDET
    cv2.imwrite("sonuc.jpg", img_cv2)
    print("\n🎉 İŞLEM BİTTİ! 'sonuc.jpg' kontrol et.")

if __name__ == "__main__":
    main()