# 🐸 Kermit

![Kermit](kermit.jpg)

**Kermit** è un pacchetto Julia che esplora l'uso del **machine learning simbolico** applicato all'**elaborazione del linguaggio naturale (NLP)**.

L'obiettivo di questo progetto è testare approcci che combinano rappresentazioni simboliche e apprendimento automatico per affrontare problemi legati alla comprensione e manipolazione del linguaggio.

## 🚀 Obiettivi

- Investigare tecniche simboliche in contesti NLP
- Sviluppare pipeline puramente simboliche con l'uso di Random Forests
- Capire limiti e potenzialità di questo approccio

## 📦 Stato del progetto

Attualmente in fase sperimentale. Il progetto si trova nelle fasi iniziali di sviluppo:

- Costruzione delle strutture base
- Testing di diverse librerie e strategie
- Implementazione di un classificatore di messaggi (Spam/Non-Spam) basato su dati generati da LLM

## 🧪 Test

I test sono disponibili nella cartella `src/`.

Per eseguirli:

```julia
using Pkg
Pkg.activate(".")  # Attiva l'ambiente del progetto
include("src/Kermit.jl")
include("src/Piggy.jl")
```

## 🔧 Installazione

```julia
using Pkg
Pkg.add(url="https://github.com/Perro2110/Kermit.jl")
```

## 👥 Contribuire

Contributi e suggerimenti sono benvenuti!
