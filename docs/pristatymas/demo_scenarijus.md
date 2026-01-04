# Demonstracijos Scenarijus
## AI Dokumentų Palyginimo Sistema

---

## 📋 Prieš Demo - Checklist

### Sisteminis Pasiruošimas

- [ ] Patikrinti, kad sistema veikia: http://localhost:7860
- [ ] Paruošti demo PDF failus (2-3 poras)
  - Skaitmeninis PDF (version A ir B su pakeitimais)
  - Skenuotas PDF (jei turite)
  - Dokumentas su lentelėmis (optional)
- [ ] Atidaryti browser tab'ą su Gradio UI
- [ ] Paruošti terminal window (jei reikės rodyti console output)
- [ ] Turėti atsarginių PDF failų (backup plan)

### Dokumentų Paruošimas

**Idealūs Demo Dokumentai**:

1. **Simple Text Changes** (pradedantiesiems)
   - 2-3 puslapių dokumentas
   - Keletas teksto pakeitimų
   - Formatavimo skirtumai (font, color)

2. **Complex Document** (advanced demo)
   - Daugiau puslapių (5-10)
   - Lentelės
   - Paveikslėliai
   - Struktūriniai pakeitimai

3. **Scanned PDF** (OCR demo)
   - Skenuotas dokumentas arba
   - PDF sukurtas iš image

---

## 🎬 Demo Scenarijus 1: Skaitmeninis PDF Palyginimas

**Trukmė**: ~5 minutės  
**Tikslas**: Parodyti pagrindinį funkcionalumą

### Žingsnis 1: Failų Įkėlimas (30s)

**Kas daryti**:
1. Atidaryti Gradio UI (http://localhost:7860)
2. Įkelti Document A (drag-and-drop arba click)
3. Įkelti Document B
4. Parodyti, kad abi failai matosi

**Kas pasakyti**:
> "Sistema priima du PDF dokumentus. Galime juos įkelti tiesiog nuvilkdami į langą. Sistema automatiškai aptinka, ar dokumentai yra skaitmeniniai ar skenuoti."

### Žingsnis 2: Parametrų Pasirinkimas (30s)

**Kas daryti**:
1. Palikti default parametrus:
   - Sensitivity: 0.82
   - Scanned Document Mode: OFF (abi PDF laikomos skenuotomis; prioritetas OCR)
   - Heatmap overlay: ON (vizualiniai skirtumai)
2. Paspausti "Compare Documents"

**Kas pasakyti**:
> "Jautrumo threshold nustato, kaip griežtai sistema aptinka skirtumus. 0.82 reiškia, kad tekstai turi būti bent 82% panašūs, kad būtų laikomi vienodais. Scanned Document Mode (abi PDF laikomos skenuotomis; prioritetas OCR) naudojamas tik tada, kai abu dokumentai yra vaizdai. Mūsų atveju dokumentai skaitmeniniai, tai jo nereikia."

### Žingsnis 3: Rezultatų Peržiūra (2 min)

**Kas daryti**:
1. Palaukti, kol sistema apdoroja (~3-10s)
2. Parodyti rezultatų santrauką:
   ```
   Comparison Results:
   - 2 pages aligned
   - 15 differences found
     • 8 content changes
     • 5 formatting changes
     • 2 layout changes
   ```
3. Scroll per Gallery View
4. Parodyti side-by-side palyginimą

**Kas pasakyti**:
> "Sistema automatiškai suderino puslapius tarp dokumentų ir rado 15 skirtumų. Matome, kad dauguma yra turinio pakeitimų, bet yra ir formatavimo skirtumų. Gallery view leidžia matyti abu dokumentus vienu metu."

### Žingsnis 4: Diff Navegacija (1.5 min)

**Kas daryti**:
1. Atidaryti "Differences Found" sąrašą
2. Click ant pirmojo diff
3. Parodyti, kaip sistema nušoka į tą vietą
4. Highlight diff puslapyje
5. Click "Next" mygtuką
6. Naršyti per kelis diff

**Kas pasakyti**:
> "Diff navigatorius parodo visus rastus skirtumus. Galime click ant bet kurio ir sistema automatiškai parodo tą vietą dokumente. Matome tikslią lokaciją, diff tipą, ir pasikeitusį turinį. Previous/Next mygtukai leidžia greitai naršyti per visus skirtumus."

### Žingsnis 5: Filtrai (1 min)

**Kas daryti**:
1. Parodyti diff filtrus:
   - Show Content ☑️
   - Show Formatting ☑️
   - Show Layout ☑️
   - Show Visual ☑️
2. Išjungti "Show Formatting"
3. Parodyti, kad diff sąrašas pasikeitė
4. Įjungti atgal

**Kas pasakyti**:
> "Filtrai leidžia pasirinkti, kokius skirtumus norime matyti. Pavyzdžiui, jei mus domina tik turinio pakeitimai, galime išjungti formatavimo ir layout skirtumus."

---

## 🎬 Demo Scenarijus 2: Skenuotas PDF su OCR

**Trukmė**: ~5 minutės  
**Tikslas**: Parodyti OCR funkcionalumą

### Žingsnis 1: Įkėlimas (30s)

**Kas daryti**:
1. Įkelti skenuotą PDF arba PDF iš image
2. Įjungti "Scanned Document Mode" ☑️
3. (Optional) Parodyti "OCR Enhancement (Hybrid, safe for digital PDFs)" (native + OCR su saugikliu; neperrašo native teksto)

**Kas pasakyti**:
> "Kai turime skenuotą dokumentą, įjungiame Scanned Document Mode. Pagal nutylėjimą sistema naudoja **PaddleOCR** (veikia CPU/Mac), o **Tesseract** yra atsarginis variantas. **DeepSeek-OCR** yra optional/guarded (įjungiamas tik sąmoningai per nustatymus ir priklauso nuo aplinkos), bet UI jis specialiai neiškeliamas kaip default pasirinkimas."

### Žingsnis 2: OCR Apdorojimas (1 min)

**Kas daryti**:
1. Paspausti "Compare Documents"
2. Parodyti, kad procesas vyksta (gali užtrukti ilgiau)
3. (Optional) Parodyti console log su OCR engine info:
   ```
   INFO: Using PaddleOCR engine for scanned document
   INFO: Processing page 1/5...
   ```

**Kas pasakyti**:
> "OCR procesas gali užtrukti ilgiau nei skaitmeninių dokumentų apdorojimas, nes sistema turi atpažinti tekstą iš paveikslėlių. Šiuo metu vyksta teksto atpažinimas su pasirinktu OCR varikliu (dažniausiai PaddleOCR), tada tekstas lyginamas kaip įprastai."

### Žingsnis 3: OCR Rezultatai (2 min)

**Kas daryti**:
1. Parodyti, kad tekstas buvo sėkmingai atpažintas
2. Parodyti bounding boxes (jei matosi)
3. Compare su kitu dokumentu

**Kas pasakyti**:
> "Sistema sėkmingai atpažino tekstą iš skenuoto dokumento. Matome, kad OCR aptiko teksto blokus, jų pozicijas, ir dabar galime palyginti su kitu dokumentu kaip įprastai."

### Žingsnis 4: OCR Engine Selection (1.5 min)

**Kas daryti**:
1. Parodyti "OCR Engine" dropdown:
   - paddle (default)
   - tesseract (fallback)
2. (Optional) Pakeisti priority ir palyginti greitį

**Kas pasakyti**:
> "Sistema palaiko kelis OCR variklius. UI leidžia pasirinkti tarp **PaddleOCR** (paddle) ir **Tesseract** (tesseract). DeepSeek-OCR projekte egzistuoja kaip optional/guarded variantas, bet nėra numatytas kaip standartinis UI pasirinkimas dėl suderinamumo tarp skirtingų mašinų."

---

## 🎬 Demo Scenarijus 3: Advanced Features

**Trukmė**: ~3-5 minutės  
**Tikslas**: Parodyti papildomas galimybes

### A. Synchronized Viewer (1.5 min)

**Kas daryti**:
1. Įjungti "Use Synchronized Viewer" ☑️
2. Parodyti premium PDF viewer
3. Naršyti per puslapius (Prev/Next)
4. Pademonstruoti sync scrolling

**Kas pasakyti**:
> "Synchronized viewer yra premium režimas, kur galime matyti abu dokumentus synchronized būdu. Abu PDF viewers sinchronizuojasi - kai scroll vieną, kitas seka automatiškai. Page navigation mygtukai leidžia šokti tarp puslapių."

### B. Heatmap overlay (1 min)

**Kas daryti**:
1. Įjungti "Heatmap overlay" ☑️
2. Palyginti dokumentus
3. Parodyti vizualinius heatmap dengimus
4. Paaiškinti spalvas (raudona = skirtumas)

**Kas pasakyti**:
> "Heatmap overlay rodo pixel-level skirtumus tarp dokumentų. Raudonos zonos rodo, kur yra vizualiniai skirtumai. Tai ypač naudinga aptikti smulkius formatavimo pakeitimus ar paveikslėlių skirtumus."

### C. Export Features (1.5 min)

**Kas daryti**:
1. Click "Export JSON"
2. Parodyti JSON failo struktūrą:
   ```json
   {
     "summary": {
       "total_diffs": 15,
       "content_changes": 8,
       ...
     },
     "diffs": [...]
   }
   ```
3. Click "Export PDF Report"
4. Parodyti sugeneruotą PDF ataskaitą

**Kas pasakyti**:
> "Rezultatus galime eksportuoti dviem formatais. JSON formatas yra skirtas mašininiam apdorojimui - galime integruoti su kitomis sistemomis. PDF ataskaita - žmogui skaitomas dokumentas su visais skirtumais."

---

## 💬 Galimi Klausimai & Atsakymai

### Techniniai Klausimai

**Q1: Kaip sistema nustato, ar tekstai yra panašūs?**

A: Naudojame Sentence Transformer modelį, kuris konvertuoja tekstą į 384-dimensional embedding vektorius. Tada skaičiuojame kosinuso panašumą (cosine similarity) tarp šių vektorių. Jei panašumas < 0.82 (threshold), tekstai laikomi skirtingais.

```python
# Supaprastinta versija
embedding_a = model.encode("Pirmas tekstas")
embedding_b = model.encode("Antras tekstas")
similarity = cosine_similarity(embedding_a, embedding_b)
if similarity < 0.82:
    → SKIRTUMAS!
```

**Q2: Kodėl naudojama keletas OCR variklių?**

A: Skirtingi OCR varikliai turi skirtingus reikalavimus:
- **PaddleOCR**: Default CPU/Mac sprendimas (stabilus atsiskaitymui)
- **Tesseract**: Universalus fallback
- **DeepSeek-OCR**: Optional (reikalauja suderinamos GPU aplinkos)

Sistema automatiškai pasirenka optimaliausią variantą pagal hardware.

**Q3: Kaip veikia layout analysis?**

A: Naudojame DocLayout-YOLO modelį, kuris aptinka 10 dokumentų elementų klasių:
- Titles (antraštės)
- Plain text (tekstas)
- Tables (lentelės)
- Figures (paveikslėliai)
- Formulas (formulės)
- Ir kt.

Modelis treniruotas su DocStructBench dataset (~300K dokumentų).

**Q4: Ar sistema veikia su non-English dokumentais?**

A: Taip! Visi komponentai palaiko multi-language:
- DeepSeek-OCR: multi-language
- PaddleOCR: 80+ kalbos
- Sentence Transformer: multi-language embeddings
- Tesseract: 100+ kalbos

**Q5: Kiek laiko užtrunka palyginimas?**

A: Priklauso nuo:
- **Skaitmeniniai PDF**: golden benchmark ~1.85s/page avg (p95 ~1.94s/page)
- **Skenuoti PDF su OCR**: priklauso nuo engine; OCR žingsnis yra brangiausias
- **Document complexity**: lentelės, paveikslėliai prideda laiko

Target yra <3s per puslapį (be OCR).

### Architektūros Klausimai

**Q6: Kodėl viskas lokaliai, o ne cloud?**

A: Trys pagrindinės priežastys:
1. **Privatumas**: medicininiai, teisiniai dokumentai negali būti siunčiami
2. **Kaštai**: cloud API yra brangu (GPT-4 Vision ~$0.01-0.03/page)
3. **Kontrolė**: pilna kontrolė modelių ir duomenų

**Q7: Kaip sistema suderina puslapius?**

A: Naudojame embedding-based page alignment:
1. Sugeneruojame embedding kiekvienam puslapiui
2. Skaičiuojame panašumus tarp visų page porų
3. Optimizuojame alignment maksimizuodami total similarity
4. Aptinkame insertion/deletion

```python
# Simplified
alignment = {}
for page_a in doc_a:
    best_match = find_best_match(page_a, doc_b)
    alignment[page_a.num] = best_match.num
```

**Q8: Ar galima naudoti su labai dideliais dokumentais?**

A: Šiuo metu sistema optimizuota iki ~60 puslapių (konfigūruojama MAX_PAGES). Didesniems dokumentams reikėtų:
- Chunk-based processing
- Async/parallel processing
- Result streaming

Tai yra future work item.

### Demo Klausimai

**Q9: Ar galite parodyti išsamesnį diff?**

A: Taip! (Click ant konkretaus diff)

Parodyti:
- Diff type (content/formatting/layout)
- Old text vs New text
- Character-level changes
- Confidence score
- Location (page, bounding box)

**Q10: Kaip sistema aptinka formatavimo skirtumus?**

A: Lyginame:
- **Font family**: Arial → Times New Roman
- **Font size**: 12pt → 14pt (threshold: 1pt)
- **Color**: RGB skirtumai (threshold: 10)
- **Style**: bold, italic, underline
- **Spacing**: line height, margins

---

## 🐛 Troubleshooting

### Galimos Problemos Demo Metu

#### Problema 1: Sistema Lėta

**Simptomai**: OCR procesas užtrunka ilgai

**Sprendimas**:
- Patikrinti, ar naudojami didesni DPI nustatymai
- Perjungti į kitą OCR engine (PaddleOCR greičiau už DeepSeek)
- Sumažinti RENDER_DPI `.env` faile

**Kas pasakyti**:
> "Matome, kad OCR apdorojimas gali užtrukti su dideliais dokumentais. Tai yra viena iš optimizacijos sričių - planuojame implementuoti paralelų puslapių apdorojimą."

#### Problema 2: UI Neresponsive

**Simptomai**: UI "užšąla" processing metu

**Sprendimas**:
- Palaukti, kol procesas baigiasi
- (Jei reikia) Restart aplikacijos

**Kas pasakyti**:
> "Ilgų procesų metu UI gali laikinai nereaguoti. Tai yra žinoma problema, kurią spręsime su async processing implementation."

#### Problema 3: Diff Nerodomi

**Simptomai**: 0 differences found, nors turėtų būti

**Sprendimas**:
- Patikrinti sensitivity threshold (gal per high)
- Sumažinti į 0.70-0.75
- Patikrinti diff filtrus

**Kas pasakyti**:
> "Jei threshold per aukštas, sistema gali neaptikti subtilių skirtumų. Galime sumažinti jautrumą."

#### Problema 4: OCR Neteisingai Atpažįsta

**Simptomai**: Blogas OCR rezultatas

**Sprendimas**:
- Perjungti į kitą OCR engine
- Patikrinti PDF kokybę (DPI)
- Force higher quality rendering

**Kas pasakyti**:
> "OCR tikslumas priklauso nuo originalaus dokumento kokybės. Galime pabandyti kitą OCR variklį arba padidinti rendering kokybę."

---

## 🎯 Demo Tips

### DO's ✅

1. **Pradėti nuo Simple**
   - Pirmiausia simple demo (2-3 puslapiai)
   - Paskui advanced features

2. **Paaiškinti Procesą**
   - Ne tik "click čia", bet ir "kodėl"
   - Susieti su teorija (embeddings, similarity)

3. **Parodyti Real Value**
   - "Įsivaizduokite, kad turite 50 puslapių sutartį..."
   - Praktiniai use cases

4. **Pripažinti Limitations**
   - "Dar dirbame ties..."
   - "Planuojame implementuoti..."

5. **Turėti Backup Plan**
   - Atsarginiai PDF failai
   - Screenshots (jei sistema neveikia)

### DON'Ts ❌

1. **Nenaudoti Per Didelių Failų**
   - 60+ puslapių gali užtrukti per ilgai demo

2. **Nepraleisti Klaidų**
   - Jei kažkas negerai, pripažinti ir paaiškinti

3. **Neskubėti**
   - Geriau lėčiau, bet aiškiai

4. **Neperkelti Per Daug Techninių Detalių**
   - Nebent klausė

5. **Neužmiršti Konteksto**
   - Ne tik "kaip", bet ir "kodėl"

---

## ⏱️ Laiko Planas (15 min pristatymas)

```
00:00-02:00  Įvadas & Sistemos Apžvalga
             └─ Kas yra sistema, kodėl svarbu
             
02:00-07:00  Demo 1: Skaitmeninis PDF
             ├─ Failų įkėlimas
             ├─ Procesing
             ├─ Rezultatų peržiūra
             └─ Diff navigation
             
07:00-11:00  Demo 2: OCR Funkcionalumas
             ├─ Scanned Document Mode
             ├─ OCR processing
             └─ Multi-engine support
             
11:00-13:00  Demo 3: Advanced Features
             ├─ Synchronized viewer
             ├─ Heatmap overlay
             └─ Export features
             
13:00-15:00  Q&A & Išvados
             └─ Klausimai, diskusija
```

---

## 📊 Success Metrics Demo

### Ką Parodyti, Kad Demo Pavyko

- ✅ Sistema veikia sklandžiai
- ✅ Visi pagrindiniai features pademonstruoti
- ✅ Auditorija supranta value proposition
- ✅ Klausimų atsakyta aiškiai
- ✅ Limitations pripažinti sąžiningai
- ✅ Ateities planai pristatyti

### Red Flags 🚩

- ❌ Per daug techninių terminų be paaiškinimų
- ❌ Demo fails ir nėra backup plano
- ❌ Neatsakyti į klausimus
- ❌ Gynybinė pozicija dėl limitations
- ❌ Praleidžiamas "kodėl" (tik "kaip")

---

## ✅ Prieš Demo - Final Checklist

**30 min prieš**:
- [ ] Sistema veikia: http://localhost:7860
- [ ] Demo PDF failai paruošti ir accessible
- [ ] Browser clean (uždaryt nereikalingus tabs)
- [ ] Presentation documents atidaryti
- [ ] Backup plan ready (screenshots)

**10 min prieš**:
- [ ] Test run simple comparison
- [ ] Patikrinti console (no errors)
- [ ] Išvalyti output directories
- [ ] Water nearby ☕

**5 min prieš**:
- [ ] Deep breath 😊
- [ ] Review key points
- [ ] Pasitikrinti audio/video (jei remote)

---

**Sėkmės su Demo! 🚀**

**Remember**: Tai prototipas, ne production sistema. Fokus ant to, ką pasiekėte, ne to, ko dar nėra.
