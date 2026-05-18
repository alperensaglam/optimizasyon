content = """# Gemini.md - Furkan Kırteke

## 🎯 Proje Rolü ve Sorumluluk
Projenin **Dış Katman (Outer Layer)** optimizasyonundan sorumlusun. Görevin, iç katmandaki CFLP (Capacitated Facility Location Problem) modelini bir "kara kutu" olarak kabul ederek, modelin performansını belirleyen $\\alpha$ ve $Q$ parametrelerini en iyileyen yöntemleri geliştirmektir.

---

## 🏗️ Matematiksel Çerçeve: Knee-Point Loss
Optimizasyon süreçlerinde kullanacağın temel amaç fonksiyonu **Knee-Point Loss** olarak tanımlanmıştır:

* **Fonksiyon Tanımı:** $L(\\alpha, Q) = \\hat{N}(\\alpha, Q) + \\hat{S}(\\alpha, Q)$.
* **$\\hat{N}$:** Normalize edilmiş açık dağıtım merkezi (DC) sayısı.
* **$\\hat{S}$:** Normalize edilmiş toplam servis maliyeti.
* **Normalizasyon:** Fonksiyon değerlerini $[0,1]$ aralığına çekmek için çalışma öncesinde belirlenen aralıklar üzerinde ön örnekleme (pre-sampling) yapman gerekmektedir.

---

## 🛠️ Uygulanacak Optimizasyon Yöntemleri
`optimization/search.py` dosyası altında şu yöntemleri geliştirmen beklenmektedir:

### 1D Arama (Sadece $\\alpha$ parametresi için)
1.  **Golden Section Search:** Her iterasyonda tek bir MILP çağrısı yaparak $L(\\alpha)$ üzerinde çalışır.
2.  **Fibonacci Search:** Golden Section ile karşılaştırmalı olarak uygulanır; fonksiyon çağrı sayısı ve doğruluk analizi yapılır.

### 2D Arama ($\\alpha$ ve $Q$ parametreleri için)
3.  **Nelder-Mead Simplex Search:** `scipy.optimize` kullanmadan; yansıma (reflection), genişleme (expansion), büzülme (contraction) ve küçülme (shrinkage) adımlarını manuel olarak kodlaman gerekmektedir.
4.  **Grid Search (Baseline):** Belirlenen aralıklarda tam tarama yaparak Nelder-Mead için bir kıyas noktası (baseline) oluşturur.

---

## 💻 Beklenen Çıktı Arayüzü (Interface)
Semih'in görselleştirme ve notebook entegrasyonu aşamasında kullanabilmesi için fonksiyonlarının şu formatta çıktı vermesi gerekmektedir:

```python
# 1D Arama
res1d = run_1d_search(solver_fn=solve_cflp, method="golden", ...)
# Beklenen: {"alpha_opt": float, "L_opt": float, "history": [{"iter": int, "alpha": float, "L": float}, ...]}

# 2D Arama
res2d = run_2d_search(solver_fn=solve_cflp, method="nelder_mead", ...)
# Beklenen: {"alpha_opt": float, "Q_opt": float, "L_opt": float, "history": [...]}