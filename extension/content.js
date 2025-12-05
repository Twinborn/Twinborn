// --- AYARLAR ---
const SERVER_URL = 'http://127.0.0.1:8000/clean-image';
// Sadece 1 sonraki resmi önceden hazırla (Sistemi yormamak için ideal)
const LOOK_AHEAD_COUNT = 1;

// --- SİSTEM DEĞİŞKENLERİ ---
let isAutoMode = false;
let processQueue = [];
let isProcessing = false;
let totalCleaned = 0;

// --- CSS STİLLERİ (HAVALI PANEL İÇİN) ---
const style = document.createElement('style');
style.innerHTML = `
  #yolo-dashboard {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: rgba(0, 0, 0, 0.9);
    color: #0f0;
    border: 1px solid #0f0;
    border-radius: 8px;
    padding: 12px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    z-index: 2147483647; /* En üstte durması için */
    box-shadow: 0 0 15px rgba(0, 255, 0, 0.2);
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: 220px;
    backdrop-filter: blur(4px);
  }
  .yolo-header { text-align: center; font-weight: bold; border-bottom: 1px solid #333; padding-bottom: 5px; margin-bottom: 5px; color: #fff; }
  .yolo-btn {
    background: #111;
    color: #fff;
    border: 1px solid #555;
    padding: 6px;
    cursor: pointer;
    font-weight: bold;
    transition: all 0.2s;
    text-transform: uppercase;
    font-size: 11px;
  }
  .yolo-btn:hover { background: #333; border-color: #fff; }
  .yolo-btn.active { background: #006400; border-color: #0f0; color: #fff; }
  .yolo-stat { display: flex; justify-content: space-between; }
  .yolo-log { font-size: 10px; color: #888; max-height: 50px; overflow: hidden; margin-top: 5px; border-top: 1px dashed #333; padding-top: 5px; }
`;
document.head.appendChild(style);

// --- KONTROL PANELİ OLUŞTURMA (UI) ---
const dashboard = document.createElement('div');
dashboard.id = 'yolo-dashboard';
dashboard.innerHTML = `
  <div class="yolo-header">MANGA AI TRANSLATOR</div>
  <div class="yolo-stat"><span>Durum:</span> <span id="yolo-status" style="color:red">BEKLEMEDE</span></div>
  <div class="yolo-stat"><span>İşlenen:</span> <span id="yolo-count" style="color:white">0</span></div>
  <div class="yolo-stat"><span>Kuyruk:</span> <span id="yolo-queue" style="color:orange">0</span></div>

  <button id="btn-toggle" class="yolo-btn">OTOMATİK MODU AÇ</button>
  <button id="btn-revert" class="yolo-btn">ORİJİNALE DÖN</button>

  <div id="yolo-msg" class="yolo-log">Sistem hazır. Başlamak için butona basın.</div>
`;
document.body.appendChild(dashboard);

// --- PANEL ETKİLEŞİMLERİ ---
const statusEl = document.getElementById('yolo-status');
const countEl = document.getElementById('yolo-count');
const queueEl = document.getElementById('yolo-queue');
const msgEl = document.getElementById('yolo-msg');
const btnToggle = document.getElementById('btn-toggle');
const btnRevert = document.getElementById('btn-revert');

function log(msg) {
    msgEl.innerText = "> " + msg;
    // Konsolu kirletmemek için logu kapattım, gerekirse açabilirsin
    // console.log("[MangaAI]:", msg);
}

btnToggle.addEventListener('click', () => {
    isAutoMode = !isAutoMode;
    if (isAutoMode) {
        statusEl.innerText = "AKTİF (TARANIYOR)";
        statusEl.style.color = "#0f0";
        btnToggle.innerText = "DURDUR";
        btnToggle.classList.add("active");
        log("Otomatik mod başlatıldı.");
        scanAndQueueVisible(); // Ekranda ne varsa hemen işle
    } else {
        statusEl.innerText = "DURAKLATILDI";
        statusEl.style.color = "orange";
        btnToggle.innerText = "OTOMATİK MODU AÇ";
        btnToggle.classList.remove("active");
        log("İşlem durduruldu. Kuyruk temizleniyor.");
        processQueue = []; // Kuyruğu boşalt
        updateStats();
    }
});

btnRevert.addEventListener('click', () => {
    if (confirm("Tüm çevirileri iptal edip orijinal sayfaya dönmek istiyor musunuz?")) {
        isAutoMode = false;
        btnToggle.classList.remove("active");
        statusEl.innerText = "SIFIRLANDI";

        const allImages = document.querySelectorAll("img[data-original-src]");
        allImages.forEach(img => {
            img.src = img.getAttribute("data-original-src");
            img.removeAttribute("data-processed");
            img.style.filter = "none";
        });
        totalCleaned = 0;
        processQueue = [];
        updateStats();
        log("Orijinal görseller geri yüklendi.");
    }
});

function updateStats() {
    countEl.innerText = totalCleaned;
    queueEl.innerText = processQueue.length;
}

// --- AKILLI İŞLEME MOTORU (BACKEND İLETİŞİMİ) ---
async function processNextInQueue() {
    if (isProcessing || processQueue.length === 0) return;
    if (!isAutoMode) return;

    // --- AKILLI FREN (Smart Brake) ---
    // Kuyruğun başındaki resme bak. Eğer kullanıcı çoktan aşağı kaydırdıysa
    // ve resim yukarıda (ekran dışında) kaldıysa, onu BOŞVER ve sıradakine geç.
    const nextImg = processQueue[0];
    const rect = nextImg.getBoundingClientRect();

    // rect.bottom < 0 demek, resim ekranın üst kısmından çıkıp gitmiş demek.
    if (rect.bottom < 0) {
        log("Hızlı geçilen resim atlandı.");
        processQueue.shift(); // Kuyruktan at
        updateStats();
        processNextInQueue(); // Bir sonrakine geç (Recursive)
        return;
    }
    // ---------------------------------

    isProcessing = true;
    const imgElement = processQueue.shift(); // İşlenecek resmi al
    updateStats();

    // Resim DOM'dan silinmişse atla
    if (!document.body.contains(imgElement)) {
        isProcessing = false;
        processNextInQueue();
        return;
    }

    await processMangaImage(imgElement);

    isProcessing = false;
    processNextInQueue(); // Zincirleme devam et
}

async function processMangaImage(imgElement) {
    // Zaten işlendiyse (veya işleniyorsa) çık
    if (imgElement.getAttribute("data-processed") === "true") return;

    // Orijinali yedekle
    if (!imgElement.getAttribute("data-original-src")) {
        imgElement.setAttribute("data-original-src", imgElement.src);
    }

    try {
        log("İşleniyor...");

        // Kullanıcıya "Burada bir şeyler oluyor" hissi ver (Sepia efekti)
        imgElement.style.transition = "filter 0.5s";
        imgElement.style.filter = "sepia(0.6) blur(1px)";

        // Resmi blob olarak al
        const response = await fetch(imgElement.src);
        const blob = await response.blob();

        // Backend'e gönder
        const formData = new FormData();
        formData.append('file', blob, "page.jpg");

        const serverResponse = await fetch(SERVER_URL, { method: 'POST', body: formData });

        if (serverResponse.ok) {
            const cleanImageBlob = await serverResponse.blob();
            const cleanImageUrl = URL.createObjectURL(cleanImageBlob);

            // Yeni resmi koy
            imgElement.src = cleanImageUrl;
            imgElement.setAttribute("data-processed", "true");
            imgElement.style.filter = "none"; // Efekti kaldır

            totalCleaned++;
            log("Çeviri Tamamlandı!");
        } else {
            log("Hata: Sunucu hatası.");
            imgElement.style.filter = "grayscale(100%)"; // Hata olursa gri yap
        }
    } catch (error) {
        // Cors hatası veya ağ hatası olursa
        log("Ağ Hatası / CORS.");
        imgElement.style.filter = "none";
    }
    updateStats();
}

// --- KAYAN PENCERE & ÖN YÜKLEME (Look Ahead) ---
const observerOptions = {
    root: null,
    rootMargin: '200px', // Ekrana girmeden 200px önce yakala
    threshold: 0.1
};

const observer = new IntersectionObserver((entries) => {
    if (!isAutoMode) return;

    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const currentImg = entry.target;

            // 1. Görünen resmi ekle
            addToQueue(currentImg);

            // 2. Bir sonraki resmi bul ve onu da ekle (Prefetch)
            let nextImg = getNextImage(currentImg);
            for (let i = 0; i < LOOK_AHEAD_COUNT; i++) {
                if (nextImg) {
                    addToQueue(nextImg);
                    nextImg = getNextImage(nextImg);
                } else {
                    break;
                }
            }

            // Motoru ateşle
            processNextInQueue();
        }
    });
}, observerOptions);

// DOM Ağacında bir sonraki <img> etiketini bulma mantığı
function getNextImage(currentElement) {
    const allImages = Array.from(document.querySelectorAll("img"));
    const currentIndex = allImages.indexOf(currentElement);
    if (currentIndex >= 0 && currentIndex < allImages.length - 1) {
        return allImages[currentIndex + 1];
    }
    return null;
}

function addToQueue(img) {
    // Kontroller: İşlenmiş mi? Zaten kuyrukta mı?
    if (img.getAttribute("data-processed") === "true") return;
    if (processQueue.includes(img)) return;

    // Gereksiz küçük resimleri (ikon, avatar) filtrele
    // Genelde manga sayfaları 300px'den geniştir.
    if (img.width < 250 || img.height < 250) return;

    processQueue.push(img);
    updateStats();
}

// --- BAŞLATMA MANTIĞI ---
function scanAndQueueVisible() {
    const allImages = document.querySelectorAll("img");
    allImages.forEach(img => observer.observe(img));
}

// Dinamik içerik (Infinite Scroll) desteği
const mutationObserver = new MutationObserver((mutations) => {
    if (!isAutoMode) return;
    mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
            if (node.tagName === 'IMG') observer.observe(node);
            else if (node.querySelectorAll) {
                node.querySelectorAll('img').forEach(img => observer.observe(img));
            }
        });
    });
});
mutationObserver.observe(document.body, { childList: true, subtree: true });