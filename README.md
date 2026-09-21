# AI Delavnica — CPU-Only Ultralytics YOLO Web Detection Demo

Cilj projekta je zgraditi preprosto, zanesljivo spletno demonstracijo detekcije objektov z **Ultralytics YOLO**, ki teče izključno na **CPU** (brez GPU/CUDA), na Linux ali WSL2 okolju.

Uporabnik v brskalniku vklopi kamero, slike se pošiljajo na lokalni FastAPI strežnik, ta izvede inferenco z YOLO26n na CPU-ju in vrne zaznane objekte, ki se izrišejo kot okvirji nazaj v brskalniku.

```text
Brskalnik (webcam)
    |  JPEG slika prek HTTP
    v
FastAPI strežnik
    |
    v
Ultralytics YOLO26n (CPU)
    |
    v
JSON z detekcijami
    |
    v
Brskalnik (izris okvirjev)
```

## Status projekta

| Del | Stanje |
|---|---|
| Specifikacija / projektna ustava (`CLAUDE.md`) | implementirano |
| FastAPI strežnik (`/health`, `/api/info`, `/api/detect`) | implementirano |
| Nalaganje modela `yolo26n.pt` prek Ultralytics | implementirano |
| Spletni odjemalec (webcam + izris okvirjev) | implementirano (ročno testiranje v brskalniku še ni bilo opravljeno) |
| OpenVINO optimizacija | ni še implementirano (načrtovano kot drugi korak) |
| Avtomatski testi | implementirano (`tests/`) |

Ta README se posodablja sproti, ko so posamezni deli implementirani.

## Zahteve

- Linux ali Ubuntu pod WSL2
- Python 3.11 ali 3.12
- Brez GPU/CUDA — celotna inferenca teče na CPU

## Hitri zagon

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Ob prvem zagonu Ultralytics samodejno prenese uteži `yolo26n.pt` (potreben je internetni dostop). Nato odpri:

```text
http://localhost:8000
```

in klikni "Start Detection" za dostop do kamere, ali uporabi polje za nalaganje slike kot alternativo.

### WSL2

Enaki ukazi kot za Linux zgoraj. Strežnik zaženeš znotraj Ubuntu instance pod WSL2, dostop iz Windows gostitelja pa poteka prek:

```text
http://localhost:8000
```

Podrobnosti o omejitvah dostopa do kamere prek brskalnika (varni kontekst, `localhost` vs. LAN dostop) bodo dokumentirane, ko bo spletni odjemalec implementiran.

### OpenVINO optimizacija (opcijsko, eksperimentalno)

OpenVINO izvoz modela je predviden kot ločen optimizacijski korak po tem, ko osnovna pot (PyTorch CPU) deluje. Ni obvezen za osnovno demonstracijo.

## Izmerjena zmogljivost

Izmerjeno na razvojnem WSL2 stroju (8 logičnih jeder), backend PyTorch CPU, `yolo26n.pt`, `imgsz=640`, testna slika 810x1080 (`bus.jpg`), prek `POST /api/detect`:

| Meritev | Vrednost |
|---|---|
| Prvi (hladni) klic | ~1800 ms |
| Ogreta inferenca (`inference_ms`, 5 klicev) | ~67–86 ms (povprečje ~76 ms) |
| Približen prepustnost (ogreto) | ~12–15 zaznav/s (serijsko, en proces) |

Meritve so bile opravljene na razvojnem/testnem stroju, ne na dejanski demo napravi — pred demonstracijo priporočamo ponovno meritev na ciljni strojni opremi.

## Arhitektura in pravila projekta

Vsa pravila glede obsega, arhitekture, prepovedanih dejanj (npr. brez treniranja modela, brez GPU, brez podatkovnih baz) in postopka dela so opisana v [`CLAUDE.md`](./CLAUDE.md), ki velja kot projektna ustava za to delo.

## Znane omejitve

- Ročno testiranje v brskalniku (dostop do kamere, izris okvirjev v živo) ni bilo opravljeno v tem koraku — preverjeno je bilo samo prek avtomatskih testov in `curl` klicev na `/api/detect`.
- OpenVINO optimizacija (Faza C iz `CLAUDE.md`) še ni implementirana; privzeta pot je PyTorch CPU.
- Meritve zmogljivosti (glej spodaj) so izmerjene na razvojnem/testnem stroju, ne na dejanski demo napravi.
