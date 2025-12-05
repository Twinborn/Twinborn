import cv2
import numpy as np
import textwrap
import easyocr
import google.generativeai as genai
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import distance
from manga_math_core import QuantumFluxLogic

# --- AYARLAR ---
GEMINI_API_KEY = "AIzaSyDsFv94D-BSxDteRJ6FRiWJr4hpojW5yxc"
TARGET_LANG = "Turkish"
FONT_PATH = "Karikatura-bold.otf" # Varsa 'animeace.ttf' veya 'comic.ttf' yap

# --- MODEL BAŞLATMA ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

print(">> [1/3] YOLO Modeli Yükleniyor (3-Sınıf: Panel, Bubble, Text)...")
yolo_model = YOLO('best.pt')

print(">> [2/3] OCR Motoru (EasyOCR) Hazırlanıyor...")
reader = easyocr.Reader(['en'])

print(">> [3/3] Gemini AI Bağlanıyor...")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-pro')

print("\n=== MANGA ENGINE V1.0 HAZIR ===")

# --- MATEMATİKSEL FONKSİYONLAR ---

def get_paper_color(img, box):
    """Kutunun içindeki baskın açık rengi (kağıt rengi) bulur."""
    x1, y1, x2, y2 = box
    h, w, _ = img.shape
    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return (255, 255, 255)

    # Köşeden örneklem al (Basit ve hızlı yöntem)
    try:
        sample = roi[5, 5]
        if np.mean(sample) < 100: return (255, 255, 255)
        return (int(sample[0]), int(sample[1]), int(sample[2]))
    except: return (255, 255, 255)

def smart_render_text(draw, text, box, bg_color, font_path=FONT_PATH):
    """
    Geliştirilmiş Render: Sığmayan metinler için 'Fallback' mekanizması eklendi.
    """
    x1, y1, x2, y2 = box
    w_box = x2 - x1
    h_box = y2 - y1

    # Çok küçük kutuları yoksay (Gürültü önleme)
    if w_box < 10 or h_box < 10:
        return

    # Font rengi seçimi
    is_dark = np.mean(bg_color) < 128
    text_color = (255, 255, 255) if is_dark else (0, 0, 0)
    stroke_color = (0, 0, 0) if not is_dark else (255, 255, 255) # Stroke zıt renk olmalı

    # Optimum font boyutunu bul
    low, high = 10, 100
    best_lines = []
    final_font = None
    final_fontsize = 10 # Varsayılan en küçük

    # 1. Aşama: Binary Search ile sığdırmaya çalış
    while low <= high:
        mid = (low + high) // 2
        try: font = ImageFont.truetype(font_path, mid)
        except: font = ImageFont.load_default()

        # Tahmini sığdırma
        avg_char_w = mid * 0.5 # Türkçe karakterler için 0.6 yerine 0.5 daha güvenli
        width_chars = max(1, int(w_box / avg_char_w))
        lines = textwrap.wrap(text, width=width_chars)

        # Ölçüm
        max_line_w = 0
        total_h = 0
        for line in lines:
            bb = draw.textbbox((0, 0), line, font=font)
            line_w = bb[2] - bb[0]
            line_h = bb[3] - bb[1]
            max_line_w = max(max_line_w, line_w)
            total_h += line_h * 1.1 # Satır boşluğu

        # Sığıyor mu? (%95 doluluk izni verelim)
        if max_line_w <= w_box * 0.95 and total_h <= h_box * 0.95:
            best_lines = lines
            final_font = font
            final_fontsize = mid
            low = mid + 1 # Daha büyüğünü dene
        else:
            high = mid - 1 # Küçült

    # 2. Aşama: FALLBACK (Eğer hiçbiri sığmadıysa)
    if not best_lines:
        # En küçük fontu al ve zorla sığdır (Gerekirse taşsın ama boş kalmasın)
        try: final_font = ImageFont.truetype(font_path, 10)
        except: final_font = ImageFont.load_default()

        # Genişliğe göre zorla böl
        avg_char_w = 10 * 0.5
        width_chars = max(5, int(w_box / avg_char_w)) # En az 5 karakterlik satır
        best_lines = textwrap.wrap(text, width=width_chars)
        final_fontsize = 10

    # 3. Aşama: Çizim
    line_height = final_fontsize * 1.2
    total_text_h = len(best_lines) * line_height
    curr_y = y1 + (h_box - total_text_h) / 2

    # Eğer metin kutudan taşıyorsa (y ekseninde), kutunun en tepesinden başlat
    if total_text_h > h_box:
        curr_y = y1 + 2

    for line in best_lines:
        bb = draw.textbbox((0, 0), line, font=final_font)
        line_w = bb[2] - bb[0]
        curr_x = x1 + (w_box - line_w) / 2

        # Güvenlik: Eğer hesaplama hatasıyla x negatif olursa düzelt
        if curr_x < x1: curr_x = x1

        draw.text((curr_x, curr_y), line, font=final_font, fill=text_color, stroke_width=2, stroke_fill=stroke_color)
        curr_y += line_height

    # Çizim (Center Alignment)
    line_height = final_fontsize * 1.2
    total_text_h = len(best_lines) * line_height
    curr_y = y1 + (h_box - total_text_h) / 2

    for line in best_lines:
        bb = draw.textbbox((0, 0), line, font=final_font)
        line_w = bb[2] - bb[0]
        curr_x = x1 + (w_box - line_w) / 2

        # Stroke (Dış Çizgi) ekleyelim ki okunabilirlik artsın (Manga stili)
        outline_color = (0,0,0) if not is_dark else (255,255,255)
        draw.text((curr_x, curr_y), line, font=final_font, fill=text_color, stroke_width=2, stroke_fill=bg_color)
        curr_y += line_height

# --- ANA ENDPOINT ---

@app.post("/clean-image")
async def clean_image_endpoint(file: UploadFile = File(...)):
    # 1. Görseli Hazırla
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. YOLO Tahmini (3 Sınıf: 0=Bubble, 1=Panel, 2=Text -- SENİN MODELİNE GÖRE)
    # Modelindeki sınıf ID'lerini kontrol et. Genelde text son sınıftır.
    results = yolo_model(img_cv)

    raw_bubbles = []
    raw_texts = []

    for box in results[0].boxes:
        cls_name = yolo_model.names[int(box.cls[0])]
        coords = list(map(int, box.xyxy[0]))

        if cls_name == 'bubble': raw_bubbles.append(coords)
        elif cls_name == 'text': raw_texts.append(coords)
        # Panel sınıfını şimdilik sıralamada kullanmıyoruz, basit tutuyoruz.

    # 3. KÜMELEME: QUANTUM FLUX LOGIC (V2)
    # Uses Cost Matrix Optimization for precise linking
    math_engine = QuantumFluxLogic()
    bubble_text_pairs = math_engine.process(raw_texts, raw_bubbles)

    print(f"Tespit: {len(bubble_text_pairs)} konuşma alanı. (Quantum Flux Engine)")

    # 5. OCR & ÇEVİRİ HAZIRLIĞI
    prompt_list = []

    for idx, pair in enumerate(bubble_text_pairs):
        cx1, cy1, cx2, cy2 = pair['clean_box']
        # Resimden kes
        crop = img_cv[cy1:cy2, cx1:cx2]

        try:
            # EasyOCR
            res = reader.readtext(crop, detail=0)
            original_text = " ".join(res)
        except: original_text = ""

        if not original_text.strip(): original_text = "..."

        pair['original'] = original_text
        prompt_list.append(f"Line {idx+1}: {original_text}")
        print(f"  > OCR [{idx+1}]: {original_text}")

    # 6. TOPLU GEMINI ÇEVİRİSİ (Context Koruma)
    translated_map = {}

    if prompt_list:
        full_prompt = f"""
        Translate these manga dialogues to {TARGET_LANG}.
        Strict Rules:
        1. Maintain the narrative flow.
        2. Output ONLY a JSON array of strings: ["Line 1 translation", "Line 2 translation"...]
        3. Do not add keys, just a flat list.

        Dialogues:
        {json.dumps(prompt_list)}
        """
        try:
            response = gemini_model.generate_content(full_prompt)
            clean_resp = response.text.replace("```json", "").replace("```", "").strip()
            translations = json.loads(clean_resp)

            # Eşleştirme
            for i, trans in enumerate(translations):
                if i < len(bubble_text_pairs):
                    bubble_text_pairs[i]['translated'] = trans
        except Exception as e:
            print(f"  ! Çeviri Hatası: {e}")
            # Hata varsa orijinali bas
            for pair in bubble_text_pairs: pair['translated'] = pair['original']

    # 7. RENDER VE MONTAJ
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    for pair in bubble_text_pairs:
        # A. Temizle (Clean Box bölgesini sil)
        cx1, cy1, cx2, cy2 = pair['clean_box']
        paper_color = get_paper_color(img_cv, pair['clean_box'])

        # Genişletilmiş silme (Güvenlik payı)
        draw.rectangle([cx1-3, cy1-3, cx2+3, cy2+3], fill=paper_color)

        # B. Yaz (Render Box içine ortala - Yani BALONUN içine)
        final_text = pair.get('translated', '...')
        smart_render_text(draw, final_text, pair['render_box'], paper_color)

    # 8. ÇIKIŞ
    res_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode('.jpg', res_img)
    return Response(content=encoded.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)