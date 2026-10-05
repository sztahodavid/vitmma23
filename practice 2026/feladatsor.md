# Gyakorlat: érzelemfelismerő rendszer nulláról demóig

**MI alapú ember-gép interakció · BME VIK**
**Emberi emóciók felismerése**

**Törzsanyag: 90 perc (0-6. blokk) · kiegészítő blokkok: +49 perc (7-9.)**

---

## Mi a gyakorlat célja

A hallgató a gyakorlat végén **egyedül is fel tud építeni egy beszédalapú
érzelemfelismerő rendszert demó szintig**. Nem versenyképes modellt építünk —
a cél az, hogy a teljes láncot egyszer végigcsinálja, és lássa, hol dől el a
teljesítmény.

Három dolgot akarunk, hogy hazavigyen:

1. **A felosztás dönt, nem a modell.** Ugyanaz a jellemzőkészlet és ugyanaz az
   SVM drámaian más eredményt ad beszélőfüggő és beszélőfüggetlen felosztással.
2. **A reprezentáció a legnagyobb egyszeri nyereség.** Kézi jellemzőkről
   önfelügyelt embeddingre váltva ugrik a teljesítmény — miközben a modell
   és a felosztás változatlan.
3. **Egy korpuszon mért eredmény nem általános képesség.** A demóban a saját
   magyar beszédükön látni fogják, hogy az angol színészi adaton tanított
   modell megbotlik.

A gyakorlat végig a **te gépeden fut**, a hallgatók párhuzamosan futtatják a
sajátjukon. Nincs önálló feladatmegoldás, de a notebook úgy van felépítve, hogy
utána egyedül is végig tudják csinálni.

---

## Amit ki kell adni — és mikor

| fájl | mikor | mit csinál |
|---|---|---|
| `ser_utils.py` | a gyakorlat előtt | segédmodul: letöltés, jellemzők, embedding, kiértékelés, mikrofon |
| `00_elokeszites.ipynb` | **legkésőbb 2 nappal előtte** | GPU-ellenőrzés, csomagok, RAVDESS a Drive-ra, WavLM a gyorsítótárba, önteszt |
| `gyakorlat.ipynb` | az óra elején | az órai notebook |
| `feladatsor.md` | az óra elején | ez a dokumentum |

A `ser_utils.py`-ban a `BASE_URL`-t és a `DRIVE_DIR`-t a `make_notebooks.py`
tetején kell átírni a tényleges letöltési helyre, majd újragenerálni a
notebookokat.

**Az előkészítő notebook futtatása nem opcionális.** A RAVDESS 208 MB, a WavLM
380 MB — ha ezt harminc ember egyszerre tölti le óra elején, elmegy húsz perc.
Érdemes a Moodle-üzenetben kiemelni, és órakezdéskor egy kézfeltartással
leellenőrizni.

---

## Blokkról blokkra

### 0 · Indulás — 5 perc

**Lépések:** Drive csatolása, `ser_utils` importálása, GPU ellenőrzése, adat
beolvasása.

**Várható kimenet:** `(1440, 10)` alakú DataFrame.

> **Buktató:** ha valakinek nincs GPU-s futtatókörnyezete, a 4. blokk
> embedding-számítása CPU-n 15–20 percig tart. Ilyenkor mondd, hogy nézze a
> vetítőt, és a `waves` listát szűkítse le (`df = df[df.actor <= 12]`).

---

### 1 · Az adat: mit is akarunk felismerni — 12 perc

**Cél:** hogy ne absztrakcióként kezeljék az adatot. Hallják és lássák.

**Lépések:**

1. Osztályeloszlás és beszélőnkénti eloszlás kirajzolása
2. Ugyanaz a színész, ugyanaz a mondat, két különböző érzelem — meghallgatás
3. Ugyanez hullámformán és spektrogramon

**Várható kimenet:** 8 kategória, mindegyikből 192 felvétel, kivéve a
semlegest: abból 96.

**Kérdések a teremhez:**

- *Miért van a semlegesből fele annyi?* → A RAVDESS-ben a semleges kategóriához
  nincs „erős” intenzitású változat. Következmény: kiegyensúlyozatlan adat,
  tehát az accuracy félrevezető, és `class_weight="balanced"` kell.
- *Mit lát egy MFCC-alapú jellemzőkészlet a két felvétel különbségéből, és mit
  nem?* → Látja a hangerőt, a tempót, az alapfrekvencia-tartományt és a
  spektrum meredekségét. Nem látja a kontextust és a szándékot.

> **Érdemes kimondani:** a két mondat szövege minden felvételen ugyanaz. Tehát
> amit a modell megtanulhat, az tisztán a *hogyan* — ez a RAVDESS legnagyobb
> előnye tanításra, és egyben a legnagyobb korlátja: a valóságban a szöveg is
> információt hordoz.

---

### 2 · Kézi jellemzők és az első modell — 15 perc

**Cél:** az előadás „alacsonyszintű jellemzők → statisztikai függvények →
beszédszintű vektor” láncának végigfuttatása, és egy szándékosan hibás
kiértékelés.

**Lépések:**

1. `load_all` — az összes felvétel memóriába (≈ 1 perc)
2. `egemaps_features` — 88 eGeMAPSv02 jellemző felvételenként (≈ 1 perc)
3. Véletlen 80/20 felosztás, SVM, kiértékelés

**Várható eredmény:** *(nagyságrend, nem garancia — lásd a próbafutásról szóló
részt lent)*

| | accuracy | makro-F1 |
|---|---|---|
| kézi jellemzők + véletlen split | ~0,55–0,70 | ~0,55–0,70 |

**Kérdés a teremhez:** *Elégedettek lennénk ezzel? Mit jelentene, ha ez a szám
egy szakdolgozatba kerülne?*

Ne áruld el a választ — a következő blokk adja meg.

> **Buktató:** az `opensmile` telepítése ritkán elhasal. A `ser_utils`
> automatikusan librosa-alapú jellemzőkre vált; a számok kicsit mások lesznek,
> a tanulság ugyanaz. Ezt érdemes szóvá tenni, ha valakinél más érték jön ki.

---

### 3 · A csapda: beszélőfüggő vs. beszélőfüggetlen — 12 perc

**Cél:** a gyakorlat legfontosabb tanulsága.

**Lépések:**

1. `speaker_split(df, test_actors=(21,22,23,24))` — a teszt színészei sosem
   szerepelnek tanításban
2. Ugyanaz a jellemzőkészlet, ugyanaz a modell, újra kiértékelés
3. A két eredmény egymás mellett oszlopdiagramon
4. Tévesztési mátrix

**Várható eredmény:**

| | accuracy | makro-F1 |
|---|---|---|
| kézi jellemzők + véletlen split | ~0,55–0,70 | ~0,55–0,70 |
| kézi jellemzők + beszélőfüggetlen | ~0,40–0,50 | ~0,38–0,48 |

A **különbség** a lényeg, nem az abszolút érték. Tíz-húsz százalékpont esést
érdemes várni.

> **Amit mondj:** „A modell nem lett rosszabb. Csak eddig a beszélőt is
> megtanulhatta a feladat mellett, most pedig nem. Ha egy cikkben nyolc
> osztályon 90% feletti eredményt láttok spontán beszéden, az első kérdés
> mindig az legyen: **hogyan osztották fel az adatot?**”

Ez a blokk kapcsolódik közvetlenül az előadás `split` ábrájához — érdemes
visszautalni rá.

---

### 4 · Önfelügyelt embedding: WavLM — 20 perc

**Cél:** annak megmutatása, hogy a reprezentáció csere önmagában mennyit ér, és
hogy nem mindegy, melyik réteget használjuk.

**Lépések:**

1. `wavlm_embeddings(waves, layers="all")` — mind a 13 rejtett réteg
   időbeli átlaga (T4 GPU-n ≈ 2 perc)
2. A 6. réteggel ugyanaz az SVM, ugyanaz a beszélőfüggetlen felosztás
3. Három kísérlet egymás mellett
4. **Rétegenkénti pásztázás:** mind a 13 rétegre külön SVM, makro-F1 kirajzolva

**Várható eredmény:**

| | accuracy | makro-F1 |
|---|---|---|
| WavLM középső réteg + beszélőfüggetlen | ~0,60–0,75 | ~0,58–0,73 |

A rétegpásztázás jellemzően **fordított U alakú**: a legjobb réteg valahol a
4–8. környékén van, a legfelső rétegek gyengébbek.

**Kérdés a teremhez:** *Miért nem a legfelső réteg a legjobb?*

→ A felső rétegek a nyelvi tartalom felé specializálódnak, mert a tanítási cél
a maszkolt keretek predikciója volt. Az érzelmet hordozó prozódiai információ a
középső rétegekben a legerősebb. Ezért használnak az éles rendszerek tanulható
súlyú rétegkombinációt egyetlen réteg helyett — pontosan ez volt az előadás
„Nem mindegy, melyik réteget használjuk” fóliája.

> **Érdemes kimondani:** a WavLM súlyait hozzá sem nyúltuk. Ez a három
> adaptációs recept közül a legolcsóbb — fagyasztott jellemzők + egy kis fej.
> Ha ennyi ilyen javulást hoz, akkor a finomhangolással érdemes csak utána
> foglalkozni.

---

### 5 · Kiértékelés rendesen — 10 perc

**Cél:** hogy ne egyetlen számmal jellemezzenek egy rendszert.

**Lépések:**

1. A két tévesztési mátrix egymás mellett
2. Osztályonkénti F1 táblázat, a változás szerint rendezve

**Amit várhatóan látni fogtok:** a semleges és a nyugodt osztály keveredik
egymással, a félelem és az undor a leggyengébb, és az SSL-reprezentáció
leginkább a közepesen nehéz osztályokon segít.

**Kérdés:** *Melyik osztályon segített legtöbbet az embedding, és melyiken alig?
Mit mond ez arról, mit tanult meg a WavLM?*

> **Fontos kalibrálás:** mondd el, hogy ez színészi, felolvasott, angol adat két
> rögzített mondattal — jóval könnyebb feladat, mint a valóság. Az előadáson
> látott Interspeech 2025 challenge spontán beszéden, nyolc osztályon **0,43
> makro-F1**-nél tartott. Ha ezt nem mondod el, a hallgatók abból indulnak ki,
> hogy a terület ott tart, ahol a mai gyakorlat.

---

### 6 · Demó: a saját hangod — 12 perc

**Cél:** a „demó szint” elérése, és egyben a korpuszfüggés megtapasztalása.

**Lépések:**

1. A legjobb beállítás újratanítása a teljes adaton
2. Mikrofonfelvétel: **angolul**, színészesen — „Kids are talking by the door”
3. Mikrofonfelvétel: **magyarul**, normál beszédstílusban

**Amit várhatóan látni fogtok:** az angol, eltúlzott felvételre néha eltalálja;
a magyar, spontán felvételre jellemzően nem, és gyakran ugyanazt a néhány
osztályt adja vissza.

> **Amit mondj:** „A modell angol, színészi, stúdióban rögzített, eltúlzott
> érzelmű beszéden tanult. A ti felvételetek minden dimenzióban eltér ettől:
> nyelv, beszédstílus, csatorna, intenzitás. Ezért gyenge — és pontosan ezért
> nem szabad egy korpuszon mért eredményt általános képességnek tekinteni.”

> **Buktató:** a mikrofonengedélyt böngészőnként kell megadni, és a Colab
> mikrofonos cellája Safariban akadozhat. Chrome-ot érdemes javasolni. Ha nem
> megy, a `su.load_wav()` bármilyen feltöltött wav fájlon működik — a hallgatók
> felvehetik telefonnal és feltölthetik.


---

### 7 · Más reprezentációk: mennyit javít egy nagyobb modell? — 25 perc

**Cél:** hogy a „reprezentáció számít" állítás ne egyetlen összehasonlításon
nyugodjon, és hogy lássák, a modellméret nem lineárisan váltódik teljesítményre.

**Lépések:**

1. Három modell ugyanazon a felosztáson: `wav2vec2-base`, `wavlm-base-plus`,
   `wavlm-large` — mindegyiknél rétegpásztázás, a legjobb réteggel mérünk
2. Oszlopdiagram a kézi jellemzős alapvonallal együtt
3. A rétegprofilok egymáson, **relatív rétegmélység** szerint — így a 12 és a
   24 rétegű modell összevethető
4. *(opcionális)* `emotion2vec` — érzelem-specifikus előtanítás

**Várható eredmény:** a `wavlm-base-plus` veri az azonos méretű
`wav2vec2-base`-t; a `wavlm-large` tovább javít, de nem háromszorosan,
pedig háromszor annyi paramétere van. A rétegoptimum a nagy modellnél
arányosan nagyjából ugyanoda esik.

**Kérdések:**

- *Miért jobb a WavLM az ugyanakkora wav2vec2-nél?* → A WavLM tanításában
  denoising és beszélőkeverés is volt, ettől a paralingvisztikai információ
  jobban megmarad. Nem a méret, hanem a tanítási cél.
- *Megéri a `large`?* → Nézzék meg a javulást a futásidő és a VRAM tükrében.
  Ez a tipikus mérnöki döntés.
- *Ha az `emotion2vec` kevés adattal is jól megy, mikor éri meg
  területspecifikus előtanítást csinálni?*

> **Buktató — idő és memória.** A `wavlm-large` 24 rétegű, 1024 dimenziós:
> T4 GPU-n ~5 perc, 4 GB-os laptop GPU-n ~15-20 perc, CPU-n nagyon sokáig.
> Ha szorít az idő, vedd ki a `MODELLEK` listából. A notebook minden modellnél
> csak a legjobb réteget tartja meg (`del Em`), hogy ne fogyjon el a memória.

> **Buktató — emotion2vec.** Nem transformers-modell, a FunASR csomag kell
> hozzá (`pip install -U funasr modelscope`). A cella `try/except`-ben van:
> ha nincs telepítve, kiírja és továbbmegy. Órán érdemes előre eldönteni,
> hogy telepíted-e.

---

### 8 · Leave-one-speaker-out: mekkora a szórás? — 12 perc

**Cél:** a 3. blokk folytatása. Ott azt láttuk, hogy a *felosztás módja*
számít; itt azt, hogy a *felosztás konkrét megválasztása* is.

**Lépések:**

1. 24 forduló: minden színész egyszer teszthalmaz
2. Átlag, szórás, legjobb és legrosszabb beszélő
3. Beszélőnkénti oszlopdiagram átlag ± szórás sávval
4. Nem szerinti bontás (a RAVDESS-ben páratlan azonosító = férfi)

**Várható eredmény:** a beszélők közti szórás jellemzően **összemérhető vagy
nagyobb**, mint a 4. blokkban mért különbség a kézi jellemzők és a WavLM között.
Ez a blokk legfontosabb üzenete.

**Kérdések:**

- *Nagyobb-e a beszélők közti szórás, mint a modellek közti különbség?* → Ha
  igen, egyetlen felosztásból mért különbség önmagában nem bizonyít semmit.
- *Beleesik-e a rögzített négy színészes eredmény a LOSO átlag ± szórás sávba?*
- *Van-e rendszeres férfi-női különbség? Az adatból jön vagy a modellből?*
  → Ez átvezet a torzítás témájára; a válasz csak célzott kísérlettel dönthető el
  (lásd a notebook végén a 2. továbbvivő ötletet).
- *Mit írnál egy cikkbe: átlagot, átlag ± szórást, vagy konfidenciaintervallumot?*

> **Futásidő:** 24 SVM-tanítás, kb. 1,5-2 perc. Nem a GPU-n fut, tehát a
> laptopokon is nagyjából ugyanennyi.

> **Buktató:** ha a 7. blokkot lefuttattátok, a notebook automatikusan az ott
> nyertes modellt használja; ha nem, a 4. blokk legjobb WavLM-rétegét. A cella
> kiírja, melyikkel dolgozik — érdemes felolvasni, hogy ne legyen zavar.

---

### 9 · Osztályok összevonása — 12 perc

**Cél:** hogy a kategórialistát **tervezési döntésként** lássák, ne adottságként.

**Lépések:**

1. A „semleges" és a „nyugodt" összevonása (8 → 7 osztály)
2. Ugyanaz a modell, ugyanaz a felosztás, újra kiértékelés
3. A két tévesztési mátrix egymás mellett
4. Osztályonkénti F1 a **nem összevont** osztályokra — javultak-e ők is?

**Várható eredmény:** a makro-F1 érezhetően javul. A nem összevont osztályok
F1-je is javulhat valamennyit, mert a modellnek nem kell két nagyon hasonló
osztály közt döntenie.

**Kérdések:**

- *Mennyi a javulásból az, hogy kevesebb osztály van?* → A véletlen találat
  12,5%-ról 14,3%-ra nő. Ez nem magyarázza az egész javulást, de részben igen.
- *A nem összevont osztályok F1-je javult-e?* → Ha igen, az a döntési határok
  tisztulása.
- *Tisztességes-e a két számot egymás mellé tenni egy cikkben?* → Csak ha
  odaírod, hány osztályról van szó. Osztályszám nélkül közölt makro-F1
  értelmezhetetlen.

> **Amit érdemes kimondani:** a valódi tanulság nem az, hogy összevonással
> jobb lesz a szám. Hanem az, hogy ha az üzleti feladat szempontjából a
> semleges és a nyugodt ugyanaz, akkor **eleve így kellett volna címkézni** —
> és akkor az annotátorok egyetértése is magasabb lett volna. Ez ugyanaz a
> gondolat, mint az előadáson a feladatszűkítésről.

---

### Zárás — 4 perc

Mi hiányzik még egy éles rendszerből: a célhoz illő adat, a feladatszűkítés, a
rétegsúlyozás, a kalibráció, a keresztkorpusz teszt és a jogi keret. A notebook
végén öt továbbvivő ötlet van — érdemes megemlíteni, hogy ezekből
szakdolgozati téma is lehet.

---

## Időzítés — B-terv, ha csúszol

A **0-6. blokk a törzsanyag**, ez fér bele 90 percbe. A **7-9. blokk
kiegészítés** (+49 perc): vagy egy második alkalommal viszed végig, vagy
otthoni feldolgozásra adod ki — mindegyik önállóan futtatható, mert a notebook
automatikusan a rendelkezésre álló legjobb reprezentációt választja.

| ha csúszol… | mit hagyj ki |
|---|---|
| 5–10 perccel | a rétegenkénti pásztázást (4. blokk 4. lépése) — mutasd csak a 6. réteget |
| 10–20 perccel | az 5. blokk osztályonkénti táblázatát, és a demóból a magyar felvételt |
| 20+ perccel | az 1. blokk spektrogramjait, és a 2. blokkban csak a véletlen splitet futtasd, a beszélőfüggetlent mutasd előre kiszámolt eredményként |
| a 7-9. blokkban | a `wavlm-large`-ot a `MODELLEK` listából, és az emotion2vec-cellát |

**Amit sosem hagyj ki:** a 3. blokkot. Ha csak egy dolog fér bele, az a
beszélőfüggő vs. beszélőfüggetlen összehasonlítás. A 8. blokk ennek a
megerősítése — ha a kettő együtt megy, az a legerősebb üzenet az egész
gyakorlatban.

---

## Technikai kockázatok

| kockázat | valószínűség | mit tegyél |
|---|---|---|
| valakinek nincs GPU-s runtime-ja | magas | órakezdéskor kérdezz rá; szűkített adathalmaz (`df[df.actor <= 12]`) |
| a Zenodo lassú vagy elérhetetlen | közepes | ezért kötelező az előkészítő notebook; tarts egy saját tükörmásolatot is |
| az `opensmile` nem települ | alacsony | automatikus visszaesés librosára, a tanulság ugyanaz |
| a Drive nem csatolódik | alacsony | futtatókörnyezet újraindítása; végső esetben letöltés a session-be |
| a mikrofon nem működik | közepes | feltöltött wav fájl a `su.load_wav()`-val |

---

## A próbafutásról

A dokumentumban szereplő **várható eredmények nagyságrendi becslések**
irodalmi értékek alapján — ebben a környezetben nem tudtam RAVDESS-t letölteni,
hogy ténylegesen lemérjem őket.

**Futtasd végig a `gyakorlat.ipynb`-et egyszer óra előtt**, és írd be ebbe a
táblázatba a tényleges számokat — a hallgatóknak sokat segít, ha tudják előre,
mit kellene látniuk, és neked is, ha valami elromlik óra közben.

| kísérlet | accuracy | makro-F1 |
|---|---|---|
| kézi jellemzők + véletlen split | | |
| kézi jellemzők + beszélőfüggetlen | | |
| WavLM 6. réteg + beszélőfüggetlen | | |
| WavLM legjobb réteg | | (réteg száma: ) |
| wav2vec2-base, legjobb réteg | | (réteg száma: ) |
| wavlm-large, legjobb réteg | | (réteg száma: ) |
| emotion2vec *(ha telepítve)* | | |
| LOSO átlag ± szórás | | |
| 7 osztály (semleges+nyugodt) | | |

Érdemes a próbafutás során azt is feljegyezni, mennyi ideig futott a
`load_all`, az `egemaps_features` és a `wavlm_embeddings` — ezekkel az órai
időzítés pontosítható.

---

## Kapcsolat az előadással

| gyakorlat | előadásfólia |
|---|---|
| 1. blokk — osztályeloszlás | „Az adat a szűk keresztmetszet, nem a modell” |
| 2. blokk — eGeMAPS | „Kézi jellemzők ma: ahol az értelmezhetőség számít” |
| 3. blokk — beszélőfüggetlen split | „A leggyakoribb hiba: beszélőfüggő felosztás” |
| 4. blokk — WavLM | „Előtanítás címke nélkül, adaptáció kevés címkével” |
| 4. blokk — rétegpásztázás | „Nem mindegy, melyik réteget használjuk” |
| 5. blokk — makro-F1, tévesztési mátrix | „Metrikák: mit mérünk és miért” |
| 6. blokk — magyar demó | „Magyar nyelven” és „Hol tart a terület valójában” |
| 7. blokk — modellösszehasonlítás | „A reprezentációt is tanuljuk” és „emotion2vec” |
| 7. blokk — rétegprofilok | „Nem mindegy, melyik réteget használjuk” |
| 8. blokk — LOSO és szórás | „További buktatók a kiértékelésben” |
| 8. blokk — nem szerinti bontás | „Mérnöki felelősség” (csoportonkénti teljesítmény) |
| 9. blokk — osztályösszevonás | „Mit kezdjen ezzel a mérnök?” (feladatszűkítés) |

---

## Adatlicenc

A RAVDESS **CC BY-NC-SA 4.0** licencű: nem kereskedelmi felhasználásra szabad,
forrásmegjelöléssel, azonos feltételekkel továbbadva. Oktatási célra rendben
van; a hallgatók figyelmét érdemes felhívni rá, hogy egy céges projektben ez a
korpusz nem használható.

Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of
Emotional Speech and Song (RAVDESS).* PLoS ONE 13(5): e0196391.
