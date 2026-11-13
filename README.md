# Systém pro automatickou analýzu motoriky – dokumentace

Verze pipeline: **2.1.1** · soubory: `app.py` (grafické rozhraní), `pipeline.py` (zpracování videa a výpočet metrik), `test_pipeline_synthetic.py` (automatické testy)

---

## 1. K čemu systém slouží

Aplikace zpracuje video, na kterém dítě pracuje s deskovou hrou (motorická hra) na zdi nebo na stole, a z pohybu rukou spočítá objektivní metriky: kolik ruka urazila, jak rychle se pohybovala, jak plynulý byl pohyb, jak pravidelné bylo tempo, jak symetricky pracovaly obě ruce a jak moc se při pohybu natahoval loket.

Postup v kostce:

1. Na videu se najdou čtyři značky v rozích desky a podle nich se obraz „narovná“ na skutečný rozměr desky 57 × 57 cm. Všechny vzdálenosti a rychlosti jsou proto v **centimetrech**, nikoli v pixelech.
2. Knihovna MediaPipe sleduje obě ruce a horní část těla.
3. Systém sám pozná začátek a konec aktivity podle toho, kdy je ruka nad deskou.
4. Z poloh ruky se po odfiltrování chyb sledování vypočítají metriky a zapíšou se do tabulky.

---

## 2. Instalace

### Požadavky (`requirements.txt`)

```
customtkinter
Pillow
opencv-contrib-python
numpy
pandas
mediapipe==<verze, která u vás běží>
shapely
openpyxl
```

Důležité poznámky:

- **`opencv-contrib-python`**, ne `opencv-python`. Detekce značek používá modul `cv2.aruco`. Nikdy neinstalujte oba balíčky současně.
- **`mediapipe` je nutné zafixovat na konkrétní verzi.** Kód používá starší rozhraní `mp.solutions`, které novější verze (např. 0.10.33) už neobsahují. Verzi zjistíte příkazem `pip show mediapipe`.
- **`openpyxl`** je volitelný, ale doporučený: bez něj se nevytvoří soubor XLSX (viz kapitola 6).
- Na Linuxu je potřeba doinstalovat `tkinter` (`sudo apt install python3-tk`); na Windows a macOS je součástí Pythonu.
- Pro opakovatelné nasazení vytvořte čisté virtuální prostředí a uložte přesné verze: `pip freeze > requirements.lock.txt`.

### Spuštění

```
python app.py
```

Složka `assets/` s ikonami aktivit (`1.png` … `10.png`) musí ležet vedle `app.py`. Bez ikon aplikace funguje, jen tlačítka nemají obrázek.

---

## 3. Požadavky na nahrávání

| Parametr | Požadavek |
|---|---|
| Formát | video **na výšku, 1080 × 1920 px (9:16)** |
| Snímková frekvence | **60 fps** (tolerance ±0,5) |
| Deska | celá deska v záběru, nehybná kamera |
| Značky | čtyři značky **ID 0, 1, 2, 3** ve slovníku **DICT_4X4_50** v rozích desky |
| Vzdálenost značek | středy značek odpovídají rohům čtverce 57 × 57 cm |
| Přípony souborů | `.mp4 .avi .mov .mkv .wmv .webm .m4v` |

Odchylka od 1080 × 1920 nebo od 60 fps video **neodmítne**, ale zapíše se do tabulky jako „Upozornění formátu“. Výsledky takového videa je potřeba posuzovat opatrně.

Pokud jsou značky natištěny jinak, upravte v `pipeline.py` konstanty `MARKER_DICTIONARY_NAME` (slovník značek) a `MARKER_CENTER_INSET_CM` (vzdálenost středu značky od rohu desky; 0 = střed značky leží přesně v rohu).

---

## 4. Práce s aplikací

**Jedno video.** Tlačítkem „Vybrat video“ zvolte soubor a klikněte na tlačítko aktivity. Při zpracování se zobrazuje živý náhled s trajektoriemi (poměr stran 9:16 se zachovává, okno lze libovolně zvětšit; celá obrazovka klávesou F11, ukončení klávesou Esc).

**Dávka (batch).** Tlačítkem „Batch: vybrat složku“ zvolte složku s videi a spusťte „Spustit batch“. Zpracují se všechna podporovaná videa ve složce, jedno po druhém. Aktivita se v dávce **nevybírá**, systém pouze rozpozná, zda pracovaly obě ruce, nebo jedna (kapitola 5.6).

**Přerušení.** Tlačítko „Přerušit analýzu“ zastaví zpracování po dokončení aktuálního snímku. V dávce se uloží výsledky už zpracovaných videí a přerušené video je označeno „Přerušeno uživatelem“.

### Aktivity (jednotlivý režim)

| ID | Název | Sledované ruce |
|---|---|---|
| 1 | Symetrické pohyby | obě |
| 2 | Rozdílné tempo | obě |
| 3 | Grafomotorika | obě |
| 4 | Kreslení tvarů | obě |
| 5 | Asymetrie a Gesta | obě |
| 6 | Asymetrie a Počítání | obě |
| 7 | Křížové souhyby | obě |
| 8 | Osmička (Pravá) | pouze pravá |
| 9 | Osmička (Levá) | pouze levá |
| 10 | Koordinované pohyby | obě |

Volba aktivity určuje jen to, které ruce se sledují. **Všechny metriky se počítají u všech aktivit stejně**, jejich interpretace ale závisí na úloze (například u „Symetrických pohybů“ je důležitá korelace rukou, u „Osmičky“ plynulost a tempo jedné ruky).

---

## 5. Jak systém zpracovává video

### 5.1 Kalibrace desky

Pokud nejsou ve videu stabilně nalezeny všechny čtyři značky, zpracování skončí hláškou o nezdařené kalibraci. Poloha značek se potvrdí z několika snímků (konstanta `MARKER_CALIBRATION_SAMPLES`), z nich se vypočítá **homografie**, tedy převod pixelů na centimetry v rovině desky. Deska je nehybná, proto se po kalibraci značky už nehledají.

### 5.2 Aktivní zóna a vymezení aktivity

- Aktivní zóna je deska zvětšená o okraj **50 px**, aby se nezahazovaly ruce těsně u hrany. Okraj slouží jen k rozpoznání aktivity; přepočet na centimetry se jím nemění.
- Aktivita **začíná**, jakmile je alespoň jedna sledovaná ruka v zóně.
- Aktivita **končí**, když jsou všechny sledované ruce mimo zónu nepřetržitě **0,25 s**. Zbytek videa se pak již nezpracovává. V jednom videu se tak analyzuje právě jedno souvislé okno aktivity, a pokud ruce zónu krátce opustí (např. dítě sáhne mimo desku), zpracování se ukončí dřív, než jste čekali.
- Snímky na konci, kdy už žádná ruka v zóně nebyla, se z aktivity odstraní.

### 5.3 Sledovaný bod ruky

Poloha ruky se nesleduje jedním pevným bodem, ale **hybridně**:

- Pokud je ukazováček **natažený**, sleduje se **špička ukazováčku**. Natažení se pozná podle úhlů prstových kloubů (PIP a DIP): zapíná se nad 145° / 140°, vypíná pod 130° / 125° (hystereze, aby bod neskákal při hraničním úhlu).
- Jinak (pěst, sevřená ruka) se sleduje **střed dlaně**, tedy průměr zápěstí a čtyř kloubů MCP.
- Přepnutí mezi bodem na prstu a v dlani se provede teprve tehdy, když nové rozhodnutí trvá **nepřetržitě alespoň 0,1 s**. Krátké výpadky prstu tak nezpůsobují zbytečná přepínání. Počet přepnutí je v tabulce (kapitola 7.9).

Špička prstu a střed dlaně jsou od sebe vzdáleny asi 8–12 cm. Při přepnutí se proto trajektorie **nikdy nepočítá přes přepnutí**: za přepnutím začíná nový souvislý úsek.

### 5.4 Odfiltrování falešných detekcí

Sledování rukou občas chybně „skočí“ (např. na jiný předmět nebo na druhou stranu desky). Systém to ošetřuje ve dvou krocích:

1. **Online (při čtení videa):** pokud by se ruka mezi dvěma přijatými snímky pohnula rychleji než **350 cm/s**, snímek se zamítne. Střed dlaně se kontroluje vždy, i při změně sledovaného bodu.
2. **Offline (po zpracování celé aktivity):** krátký úsek (do **0,5 s**) oddělený od okolí nemožnými skoky, jehož sousedé na sebe navazují, je falešný „výlet“ a odstraní se. Skutečné přemístění ruky (např. po výpadku delším než pár snímků) se zachová; trajektorie se jen rozdělí na dva úseky.

Počet zamítnutých snímků je vidět v tabulce jako „Zamítnuté snímky (falešné skoky)“.

### 5.5 Doplnění výpadků a vyhlazení

- **Krátké výpadky** (do 0,2 s) mezi dvěma platnými snímky se lineárně doplní, ale jen pokud je před výpadkem i po něm sledován stejný bod a ruka nebyla mezitím zachycena mimo zónu. Delší výpadky se nedoplňují; trajektorie se v nich přeruší.
- **Vyhlazení polohy** je lokální polynom 3. stupně v časovém okně 0,3 s (typ Savitzky–Golay). Je symetrické, tedy **bez časového zpoždění**, a nezávisí na snímkové frekvenci. Odstraňuje drobný šum landmarků, ale nepřidává fázový posun, který by zkresloval rychlosti a plynulost.
- Úseky se vyhlazují a derivují **odděleně** (po každém přepnutí bodu nebo přerušení vzniká nový úsek).

### 5.6 Typ aktivity (v dávce)

Systém rozhodne podle podílu snímků, ve kterých je ruka v zóně:

| Podmínka | Typ aktivity |
|---|---|
| obě ruce alespoň v 50 % snímků a zároveň obě společně alespoň ve 25 % snímků | **Obouruční** |
| pouze levá ruka alespoň v 50 % snímků (pravá méně než v 50 %) | **Jednoruční** (levá) |
| pouze pravá ruka alespoň v 50 % snímků (levá méně než v 50 %) | **Jednoruční** (pravá) |
| jinak | **Neurčeno** |

U jednoruční aktivity se metriky druhé ruky ani společné metriky nevyplňují.

### 5.7 Určení levé a pravé ruky

MediaPipe označuje ruce „levá/pravá“ za předpokladu zrcadlového snímku. Pokud nejsou vaše videa zrcadlená, ponechte `INPUT_VIDEO_IS_MIRRORED = False` (štítky se prohodí). **Před prvním ostrým použitím ověřte na jednom videu**, že „L“ (v grafu trajektorie modrá) je opravdu levá ruka dítěte, a „R“ (žlutá) pravá. Podle toho se odvíjejí všechny metriky pro jednotlivé ruce, asymetrie a překročení středové linie.

---

## 6. Výstupy

### 6.1 Jedno video

Vedle videa vznikne složka `<název_videa>_out` s těmito soubory:

| Soubor | Obsah |
|---|---|
| `data.json` | všechny metriky a metadata (struktura odpovídá slovníku v kapitole 7) |
| `trajectory.json` | poloha ruky v každém snímku (zdrojový bod, surová poloha, zdroj prst/dlaň) |
| `trajectory_plot.png` | trajektorie rukou nakreslené na prvním snímku (levá modře, pravá žlutě; zelený obrys = deska, žlutooranžový = aktivní zóna) |
| `annotated.mp4` | video aktivity s vykreslenými body a trajektoriemi |

### 6.2 Dávka

Ve vybrané složce vzniknou:

| Soubor | Obsah |
|---|---|
| `batch_results.xlsx` | **doporučený** přehled: čísla jsou skutečné číselné buňky, nezávislé na regionálním nastavení počítače; lze řadit a počítat |
| `batch_results.csv` | stejná tabulka, středník jako oddělovač, UTF-8 s BOM, **desetinná tečka** (vhodné pro pandas/R) |
| `batch_errors.log` | podrobnosti o videích, která selhala (vzniká jen při chybě) |

Pokud je soubor zamčený (otevřený v Excelu), uloží se výsledek pod názvem s časovým razítkem.

**Rozložení tabulky:** jeden sloupec na video; řádky `Kategorie | Oblast | Metrika | Jednotka | <video 1> | <video 2> | …`. Kategorie: *R ruka*, *L ruka*, *společné metriky*, *kvalita dat*, *obecné parametry videa*. Prázdná buňka znamená, že hodnotu nebylo možné spolehlivě spočítat, nebo že se na danou aktivitu nevztahuje (např. druhá ruka u jednoruční aktivity).

> **Pozor na CSV a Excel.** Otevřete-li CSV dvojklikem, může tabulkový procesor desetinnou tečku přečíst podle místního nastavení špatně (např. „10,217“ jako 10217). Proto pro čtení používejte `batch_results.xlsx`. Chcete-li CSV s desetinnou čárkou, nastavte v `pipeline.py` `CSV_DECIMAL_SEPARATOR = ","`; zvolte to ale jen tehdy, když víte, v jakém prostředí se bude soubor otevírat.

---

## 7. Slovník metrik

Všechny metriky se počítají zvlášť pro **pravou (R)** a **levou (L)** ruku, pokud není uvedeno jinak. Jednotky platí pro kalibrované video (cm); u nekalibrovaného videa by byly v pixelech (u normálního průběhu nenastává).

### 7.1 Trajektorie

| Metrika | Jednotka | Význam |
|---|---|---|
| **Délka trajektorie** | cm | Součet vzdáleností mezi sousedními snímky vyhlazené dráhy ruky. Říká, **kolik pohybu ruka vykonala**. Počítá se jen z platných, souvislých úseků; nedoplněné výpadky se do délky nezapočítávají. |

Poznámka: i u zcela klidné ruky zůstává drobný šum, který délku mírně zvyšuje (řádově několik cm za 5 s). U krátkých a málo pohyblivých aktivit je tato nepřesnost relativně větší.

### 7.2 Rychlost

Rychlost je velikost změny polohy za čas, počítaná z vyhlazené dráhy (cm/s).

| Metrika | Význam |
|---|---|
| **Maximální rychlost** | Největší okamžitá rychlost. Je citlivá na jednotlivé chyby sledování; hodnota blízko limitu 350 cm/s je důvodem ke kontrole videa. |
| **Průměrná rychlost** | Průměr okamžitých rychlostí přes všechny snímky s platným bodem (včetně okamžiků, kdy ruka stála). |
| **95. percentil rychlosti** | Rychlost, kterou ruka překročila jen v 5 % času. Stabilnější než maximum; vhodná pro porovnávání rychlých částí pohybu. |

### 7.3 Loket

Úhel v loketním kloubu se počítá z trojice bodů **rameno – loket – zápěstí** pomocí **trojrozměrných** souřadnic těla (`pose_world_landmarks`), takže ho neovlivňuje úhel pohledu kamery. **180° = plně natažená paže**, menší úhel = pokrčenější loket.

| Metrika | Jednotka | Význam |
|---|---|---|
| **Maximální extenze** | ° | Největší úhel lokte během aktivity (jak moc se paže natáhla). |
| **Rozsah pohybu (ROM)** | ° | Rozdíl mezi největším a nejmenším úhlem (kolik se loket během aktivity „ohýbal“). |

Podrobnosti: úhel se zaznamenává jen tehdy, když je příslušná ruka v zóně a všechny tři klouby jsou dostatečně viditelné (viditelnost ≥ 0,45). Před určením maxima a rozsahu se řada úhlů očistí **mediánovým filtrem** (okno 0,12 s), aby jediný chybný snímek neurčil celý výsledek. Lze přepnout na starší úhel v rovině obrazu konstantou `ELBOW_ANGLE_MODE = "image_2d"`; hodnoty obou režimů nejsou srovnatelné.

### 7.4 Plynulost

Plynulost popisuje, jak „hladký“ je pohyb. Hladký pohyb má málo prudkých změn zrychlení. Veličinou je **jerk**, tedy změna zrychlení v čase (třetí derivace polohy, cm/s³).

| Metrika | Jednotka | Význam |
|---|---|---|
| **Normovaný jerk** | – (bez jednotky) | Bezrozměrná míra plynulosti. **Nižší hodnota = plynulejší pohyb.** |
| **Maximální jerk** | cm/s³ | Největší hodnota jerku. Velmi citlivá na šum a krátké chyby; slouží spíš jako kontrola než jako ukazatel. |
| **95. percentil jerku** | cm/s³ | Hodnota jerku, kterou pohyb překročil jen v 5 % času. Stabilnější než maximum; doporučená pro porovnávání. |

**Jak se jerk počítá.** Z vyhlazené dráhy se v každém bodě lokálním polynomem 4. stupně v okně 0,3 s odhadne třetí derivace. První a poslední 0,15 s každého souvislého úseku se **vynechá**, protože tam by odhad nebyl spolehlivý. Díky tomu přepnutí mezi bodem na prstu a v dlani ani začátek a konec úseku nevytvoří falešný skok jerku.

**Jak se počítá normovaný jerk.** Pohyb se rozdělí na **jednotlivé tahy**: tah je souvislý úsek, kdy je rychlost nad prahem klidu (5 % 95. percentilu rychlosti úseku, nejméně 2 cm/s), s délkou alespoň 0,25 s a dráhou alespoň 2 cm. Souvislý pohyb delší než 2 s (například plynulé kroužení bez pauzy) se rozdělí na stejné části. Pro každý tah platí

> NJ = √( ½ · ∫ jerk² dt · T⁵ / L² )

kde *T* je doba tahu a *L* jeho dráha. Výsledná hodnota je **medián přes všechny tahy**. Podle zdroje hodnoty uvádí tabulka v kategorii „kvalita dat“:

- **„Medián přes tahy“**: standardní, srovnatelná hodnota.
- **„Celý úsek (orientační)“**: v celé aktivitě nebyl nalezen jediný úplný tah, hodnota je spočítána z celého souvislého úseku. **Nelze ji srovnávat** s hodnotami z tahů, protože takto vypočtený normovaný jerk roste s počtem a délkou pohybů.

**Jak hodnotě rozumět.** Normovaný jerk je relativní měřítko. V syntetickém testu vychází ideálně hladký pohyb z bodu do bodu zhruba **10**; v testovací sadě dětských videí se hodnoty pohybovaly zhruba mezi 2 a 105, běžně 10–60. Porovnávejte **jen výsledky ze stejné verze pipeline** (nastavení vyhlazení a segmentace tahů hodnotu ovlivňuje), a pokud možno stejné aktivity.

### 7.5 Rytmus

| Metrika | Jednotka | Význam |
|---|---|---|
| **Variabilita tempa (CV)** | % | Jak nepravidelné bylo tempo opakovaného pohybu. **Nižší = pravidelnější tempo.** |
| **Počet pohybových period** | ks | Kolik časových intervalů mezi opakováními pohybu bylo pro výpočet použito. |

**Jak se počítá.** Dráha ruky se promítne na směr největšího pohybu, odstraní se pomalý posun (drift) a signál se lehce vyhladí (0,08 s). Pak se najdou vrcholy, tedy opakované „výchylky stejným směrem“ (nejméně 0,25 s od sebe a dostatečně výrazné). Doby mezi sousedními vrcholy jsou **periody** a

> CV = směrodatná odchylka period / průměr period × 100

Hodnota se spočítá, jen když jsou nalezeny alespoň **dva intervaly** (tři vrcholy). Je to obecná heuristika pro opakovaný pohyb, nikoli měření specifické pro konkrétní hru.

**Spolehlivost.** CV z méně než **4** intervalů je jen orientační (systém to uvede v upozornění). U pohybu, který není opakovaný (jednorázové kreslení, gesto), CV nedává smysl nebo zůstane prázdné.

**Variabilita tempa obou rukou (CV)** (společná metrika) je průměr CV pravé a levé ruky. Vyplní se **jen tehdy, když CV existuje u obou rukou**; u jednoruční aktivity zůstává prázdná.

### 7.6 Prostorová přesnost

| Metrika | Hodnoty | Význam |
|---|---|---|
| **Překročení středové linie** | Ano / Ne | „Ano“, pokud dráha ruky zasahuje **na obě strany** svislé středové čáry desky (x = 28,5 cm), a to alespoň o **1 cm** na každou stranu. Rezerva 1 cm zabrání tomu, aby drobné chvění na čáře bylo vykázáno jako překročení. |

Metrika nerozlišuje, ze které strany ruka začala; říká pouze, zda se pohybovala v obou polovinách desky. Závisí na správném určení levé a pravé ruky (kapitola 5.7).

### 7.7 Společné metriky (jen u obouručních aktivit)

| Metrika | Jednotka | Význam |
|---|---|---|
| **Rozdíl délek trajektorií** | cm | \|L − R\| (rozdíl drah levé a pravé ruky). |
| **Asymetrie délek trajektorií** | % | \|L − R\| / průměr(L, R) × 100. 0 % = obě ruce urazily stejnou dráhu. |
| **Korelace rychlostí** | r (−1 až 1) | Pearsonova korelace okamžitých rychlostí obou rukou. Blízko **1** = ruce zrychlují a zpomalují současně (synchronní tempo); blízko 0 = bez vztahu. |
| **Korelace pohybu v ose X** | r | Korelace vodorovné složky rychlosti. **Záporná** = ruce se ve vodorovném směru pohybují proti sobě (zrcadlově souměrný pohyb, např. obě ruce od středu nebo ke středu); **kladná** = obě ruce jdou stejným směrem. |
| **Korelace pohybu v ose Y** | r | Totéž pro svislou složku rychlosti (kladná = obě ruce nahoru a dolů zároveň). |
| **Variabilita tempa obou rukou (CV)** | % | Viz 7.5. |

Korelace se počítá jen ze snímků, kdy jsou platné **obě** ruce, a nejméně ze 3 snímků. Jsou-li ruce zhruba nehybné (nulový rozptyl), korelace zůstane prázdná.

### 7.8 Obecné parametry videa

| Metrika | Význam |
|---|---|
| **Typ aktivity** | Obouruční / Jednoruční / Neurčeno (kapitola 5.6). |
| **Doba aktivity** | Čas od prvního do posledního snímku zpracovaného okna aktivity, v sekundách (ne délka celého videa). |
| **Rozlišení, Orientace, Poměr stran, Snímková frekvence** | Skutečné vlastnosti dekódovaného videa. Správně: 1080 × 1920, na výšku, 9:16, 60 fps. |
| **Jednotky souřadnic / rychlosti / jerku, Metoda kalibrace** | `cm`, `cm/s`, `cm/s³` a `homografie_57cm` znamená, že deska byla úspěšně zkalibrována. |
| **Upozornění formátu** | Odchylka od očekávaného formátu videa; „Bez upozornění“ = vše v pořádku. |
| **Stav zpracování** | `OK`, popis důvodu (např. nenalezená aktivita, selhala kalibrace), `CHYBA: …`, nebo `Přerušeno uživatelem`. |
| **Verze pipeline** | Verze kódu, kterým byl výsledek vytvořen; slouží k dohledatelnosti a ke správnému porovnávání. |

### 7.9 Kvalita dat

Tato sekce říká, **jak moc výsledkům u daného videa věřit**. Každá ruka má vlastní hodnoty (u jednoruční aktivity jen aktivní ruka).

| Metrika | Význam |
|---|---|
| **Podíl snímků s platným bodem** (%) | V jaké části okna aktivity byla ruka sledována (včetně doplněných krátkých výpadků). |
| **Zamítnuté snímky (falešné skoky)** (%) | Podíl snímků vyřazených při odfiltrování falešných detekcí (kapitola 5.4). |
| **Přepnutí prst/dlaň** (ks) | Kolikrát se během aktivity změnil sledovaný bod. |
| **Zdroj normovaného jerku** | „Medián přes tahy“ (standard), nebo „Celý úsek (orientační)“ (kapitola 7.4). |
| **Spolehlivost výsledků** | **Dobrá** nebo **Ke kontrole**. |
| **Upozornění ke kvalitě dat** | Slovní seznam konkrétních důvodů, případně „Bez upozornění“. |

**Kdy je výsledek „Ke kontrole“.** Stačí splnění kterékoli z podmínek:

| Podmínka | Práh |
|---|---|
| ruka byla sledována jen v malé části aktivity | platný bod v < **70 %** snímků |
| příliš mnoho snímků zamítnuto jako falešné skoky | > **3 %** snímků |
| časté přepínání prst/dlaň | více než **1 přepnutí za sekundu** |
| maximální rychlost je u fyzikálního limitu | > **90 %** limitu, tedy > 315 cm/s |

Pouze **informační** upozornění (nemění hodnocení na „Ke kontrole“): CV tempa z méně než 4 intervalů, a normovaný jerk z celého úseku. Prahy jsou v `pipeline.py` v konstantách `QA_*` a odhadují se; po zpracování první větší sady dat je podle skutečného rozložení doladíte.

„Ke kontrole“ neznamená, že jsou data nepoužitelná. Znamená to, že se vyplatí **projít video `annotated.mp4`**, zda sledování odpovídá skutečnosti.

---

## 8. Omezení a doporučení pro interpretaci

- **Jedna kamera, rovinná kalibrace.** Poloha ruky se přepočítává do roviny desky. Střed dlaně je ale několik cm před deskou a při šikmém pohledu kamery se promítá s malou paralaxou (řádově jednotky cm podle vzdálenosti od desky). Špička prstu dotýkajícího se desky je přesnější než dlaň.
- **Chyby sledování.** MediaPipe může ruku ztratit nebo zaměnit levou a pravou. Systém falešné skoky odstraňuje, ale ne dokonale; kontrolní ukazatele jsou v sekci „kvalita dat“.
- **Krátké a nepravidelné aktivity.** Rytmus (CV) vyžaduje opakovaný pohyb; plynulost vyžaduje alespoň jeden úplný tah. U velmi krátkých videí (několik sekund) jsou některé hodnoty prázdné nebo orientační.
- **Porovnávání.** Hodnoty srovnávejte jen mezi videi zpracovanými stejnou verzí pipeline (sloupec „Verze pipeline“) a při stejném formátu videa. Absolutní hodnoty (zejména jerk a normovaný jerk) jsou závislé na způsobu vyhlazení.
- **Interpretace.** Metriky jsou popisné, objektivní údaje o pohybu. Bez referenčních dat (normy pro věk a typ úlohy) je nelze samy o sobě číst jako klinický závěr.

---

## 9. Konfigurace

Nastavení je v horní části `pipeline.py`. Nejdůležitější konstanty:

| Konstanta | Výchozí | Význam |
|---|---|---|
| `BOARD_SIDE_CM` | 57,0 | Strana desky v cm (rozměr mezi středy značek) |
| `MARKER_DICTIONARY_NAME` | `DICT_4X4_50` | Slovník tištěných značek |
| `MARKER_CENTER_INSET_CM` | 0,0 | Vzdálenost středu značky od rohu desky |
| `ACTIVITY_ZONE_MARGIN_PX` | 50 | Okraj aktivní zóny kolem desky (px) |
| `EXIT_CONFIRM_SECONDS` | 0,25 | Jak dlouho musí být ruce mimo zónu, aby aktivita skončila |
| `INPUT_VIDEO_IS_MIRRORED` | False | Zda jsou videa zrcadlená (ovlivňuje levou/pravou) |
| `MAX_MEASURED_POINT_SPEED_CM_S` | 350 | Fyzikální limit rychlosti; rychlejší skok = falešná detekce |
| `JUMP_MAX_EXCURSION_SECONDS` | 0,5 | Nejdelší falešný „výlet“, který se odstraní |
| `SOURCE_SWITCH_MIN_SECONDS` | 0,1 | Doba, po kterou musí rozhodnutí prst/dlaň trvat |
| `INTERPOLATE_GAP_LIMIT_SECONDS` | 0,2 | Nejdelší výpadek, který se doplní |
| `SMOOTHING_WINDOW_SECONDS` | 0,3 | Okno vyhlazení polohy |
| `JERK_WINDOW_SECONDS` | 0,3 | Okno pro výpočet jerku |
| `NORM_JERK_MODE` | `per_stroke` | Normovaný jerk po tazích (`per_run` = starší výpočet z celého úseku) |
| `ELBOW_ANGLE_MODE` | `world_3d` | Typ úhlu lokte (`image_2d` = starší 2D) |
| `CENTER_LINE_MARGIN_CM` | 1,0 | Rezerva pro „překročení středové linie“ |
| `CSV_DECIMAL_SEPARATOR` | `.` | Desetinný oddělovač v CSV |

Po změně konstant, které ovlivňují výpočet, se změní i hodnoty výsledků. Doporučujeme proto zvýšit `PIPELINE_VERSION`, aby se výsledky z různých nastavení nemíchaly.

---

## 10. Testy

```
python test_pipeline_synthetic.py
```

Test pracuje se syntetickými daty (bez videa) a ověřuje mimo jiné:

- přepnutí mezi bodem na prstu a v dlaň nevytváří falešný skok jerku,
- falešné skoky (jednosnímkové, vícesnímkové, na hranici přepnutí a na začátku záznamu) se odstraní,
- normovaný jerk nezávisí na počtu tahů,
- filtr přepínání prst/dlaň potlačí krátké kmitání,
- rozložení a obsah tabulky (CSV i XLSX), sekce kvality dat a zpracování zamčeného souboru.

Spouští se ve složce s `pipeline.py`. Test neověřuje samotnou detekci rukou ani značek; pro tu je potřeba vyzkoušet reálná videa.

---

## 11. Řešení problémů

| Příznak | Pravděpodobná příčina a řešení |
|---|---|
| „Nepodařilo se zkalibrovat herní zónu“ | Ve videu nejsou stabilně vidět všechny čtyři značky ID 0–3, nebo neodpovídá slovník. Zkontrolujte záběr, osvětlení a `MARKER_DICTIONARY_NAME`. |
| „V rozšířené herní zóně nebyla zachycena žádná sledovaná ruka“ | Ruce nejsou nad deskou, nebo jsou mimo záběr. |
| Zpracování skončilo dřív než video | Obě ruce byly mimo zónu déle než 0,25 s (kapitola 5.2). |
| Chyba `module 'mediapipe' has no attribute 'solutions'` | Nainstalována nová verze MediaPipe; použijte starší verzi, se kterou aplikace běží. |
| Chyba `cv2.aruco` není k dispozici | Nainstalujte `opencv-contrib-python` (a odinstalujte `opencv-python`). |
| Doba aktivity v desetitisících, čísla jsou textem | CSV otevřené s jiným desetinným oddělovačem. Použijte `batch_results.xlsx`. |
| Soubor XLSX nevznikl | Chybí `openpyxl` (`pip install openpyxl`); zůstane jen CSV. |
| „Spolehlivost: Ke kontrole“ | Viz kapitola 7.9; prohlédněte `annotated.mp4` a kvalitu sledování. |
| Levá a pravá ruka jsou prohozeny | Přepněte `INPUT_VIDEO_IS_MIRRORED` (kapitola 5.7). |
| Video skončilo stavem `CHYBA: …` | Podrobnosti jsou v `batch_errors.log` ve složce s videi. |

---

## 12. Sestavení instalačního balíčku pro Windows

Inno Setup sám Python nepřekládá, pouze vytvoří instalátor ze souborů, které mu dodáte. Aplikaci je potřeba nejdřív „zabalit“ nástrojem **PyInstaller** a výslednou složku pak vložit do instalátoru.

> Postup níže vychází z analýzy kódu. Sestavení a instalaci **vždy vyzkoušejte na čistém počítači bez Pythonu** (nebo ve virtuálním stroji), než balíček předáte uživatelům.

### 12.1 PyInstaller

Spouštějte ve **Windows**, v čistém virtuálním prostředí se stejnou verzí Pythonu a stejnými balíčky, na kterých aplikaci vyvíjíte:

```
pip install pyinstaller
pyinstaller --noconfirm --clean --windowed --name MotorAnalysis ^
  --collect-all mediapipe --collect-all customtkinter --collect-all shapely ^
  --hidden-import openpyxl --add-data "assets;assets" app.py
```

- **Režim složky (`--onedir`, výchozí), ne `--onefile`.** Jednosouborová varianta se při každém startu rozbaluje (pomalý start) a častěji ji falešně označují antiviry.
- `--collect-all mediapipe` zahrne modely a datové soubory MediaPipe; `--collect-all customtkinter` motivy a obrázky rozhraní; `--collect-all shapely` knihovny GEOS.
- `--add-data "assets;assets"` přidá ikony aktivit (na Windows se v tomto zápisu odděluje středníkem). Aplikace si je najde přes `sys._MEIPASS`.
- `--hidden-import openpyxl` zajistí, že vznikne i soubor XLSX. Pokud by v balíčku `openpyxl` chyběl, aplikace tiše vytvoří jen CSV, proto to po sestavení zkontrolujte.
- Výsledek je ve složce `dist\MotorAnalysis\` (spouštěcí soubor `MotorAnalysis.exe`).

### 12.2 Inno Setup

Do instalátoru vložte **celou** složku `dist\MotorAnalysis\`. Minimální skript:

```
[Setup]
AppName=Motor Analysis
AppVersion=2.0
DefaultDirName={autopf}\MotorAnalysis
DefaultGroupName=Motor Analysis
OutputBaseFilename=MotorAnalysis_Setup
Compression=lzma2
; 64bitová instalace (Inno Setup 6.3 a novější; u starších verzí použijte "x64")
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "dist\MotorAnalysis\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Motor Analysis"; Filename: "{app}\MotorAnalysis.exe"
Name: "{autodesktop}\Motor Analysis"; Filename: "{app}\MotorAnalysis.exe"
```

Aplikace nezapisuje do instalační složky: výsledky ukládá vedle videí a protokol chyb do domovské složky uživatele, takže stačí běžná instalace do Program Files.

### 12.3 Kontrolní seznam po sestavení

1. Aplikace se spustí na čistém počítači bez Pythonu; ikony aktivit jsou vidět.
2. Zpracování videa dokončí celý průběh (detekce značek, ruce, výstupy).
3. Test s videem ve složce, jejíž název nebo cesta obsahuje **diakritiku** (např. `C:\Users\Šárka\Videa`), dokončí export včetně `annotated.mp4` a `trajectory_plot.png`.
4. Dávka vytvoří `batch_results.xlsx` i `batch_results.csv`.
5. Při chybě vznikne `MotorAnalysis_error.log` v domovské složce uživatele (`C:\Users\<jméno>\`).
6. Antivirus (včetně Windows Defenderu) instalátor ani aplikaci neblokuje; případně instalátor podepište certifikátem.

---

