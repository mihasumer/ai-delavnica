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
| FastAPI strežnik (`/health`, `/api/info`, `/api/detect`) | ni še implementirano |
| Nalaganje modela `yolo26n.pt` prek Ultralytics | ni še implementirano |
| Spletni odjemalec (webcam + izris okvirjev) | ni še implementirano |
| OpenVINO optimizacija | ni še implementirano (načrtovano kot drugi korak) |
| Avtomatski testi | ni še implementirano |

Ta README se bo posodabljal sproti, ko bodo posamezni deli implementirani. Trenutno repozitorij vsebuje samo specifikacijo (`CLAUDE.md`), po kateri se vodi implementacija.

## Zahteve

- Linux ali Ubuntu pod WSL2
- Python 3.11 ali 3.12
- Brez GPU/CUDA — celotna inferenca teče na CPU

## Hitri zagon (načrtovano, še ni implementirano)

Ko bo osnovna implementacija (Faza A/B iz `CLAUDE.md`) končana, bo zagon potekal takole:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### WSL2

Enaki ukazi kot za Linux zgoraj. Strežnik zaženeš znotraj Ubuntu instance pod WSL2, dostop iz Windows gostitelja pa poteka prek:

```text
http://localhost:8000
```

Podrobnosti o omejitvah dostopa do kamere prek brskalnika (varni kontekst, `localhost` vs. LAN dostop) bodo dokumentirane, ko bo spletni odjemalec implementiran.

### OpenVINO optimizacija (opcijsko, eksperimentalno)

OpenVINO izvoz modela je predviden kot ločen optimizacijski korak po tem, ko osnovna pot (PyTorch CPU) deluje. Ni obvezen za osnovno demonstracijo.

## Arhitektura in pravila projekta

Vsa pravila glede obsega, arhitekture, prepovedanih dejanj (npr. brez treniranja modela, brez GPU, brez podatkovnih baz) in postopka dela so opisana v [`CLAUDE.md`](./CLAUDE.md), ki velja kot projektna ustava za to delo.

## Znane omejitve

- Implementacija (FastAPI strežnik, spletni odjemalec, testi) še ne obstaja — trenutno je na voljo samo specifikacija.
- Merjenja zmogljivosti (latenca, FPS) bodo objavljena šele, ko bo obstajala delujoča implementacija za testiranje na dejanski strojni opremi.
