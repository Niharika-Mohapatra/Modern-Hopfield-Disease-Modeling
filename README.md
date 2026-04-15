# Modelling Neurodegeneration with Modern Hopfield Networks

---

## Overview

This project uses a **Modern Hopfield Network** to simulate the progressive memory failure seen in neurodegenerative diseases such as Alzheimer's. Greyscale images of Asian country flags serve as memory patterns. The network's ability to retrieve a corrupted flag represents a brain's ability to recall a degraded memory.

Three biologically-motivated stages of decline are modelled sequentially:

| Stage | Biological Basis | Model Mechanism |
|-------|-----------------|-----------------|
| **Stage 1** | Astrocyte dysfunction | Glutamate neuromodulation (`g`) dysregulates, altering retrieval sharpness and step size |
| **Stage 2** | Synaptic pruning | Memory patterns are structurally corrupted via per-synapse dropout |
| **Stage 3** | Neuronal death | Neurons are randomly silenced; active neurons face higher deletion probability (excitotoxicity) |

---

## Repository Structure

```
├── hopfield_neurodegeneration.ipynb   # Main Kaggle notebook
├── hopfield_networks_utils.py         # Base Hopfield utilities (auto-downloaded)
└── README.md
```

---

## Dataset

**Flags of Asia** — greyscale images scraped by me from Wikipedia:
[nmohapatra/flags-of-asia-123](https://www.kaggle.com/datasets/nmohapatra/flags-of-asia-123)

Pixel values are scaled to `[-1, 1]` before use.

---

## Model

### Modern Hopfield Network

The energy function is:

$$E = -\text{lse}(\beta,\ \Xi^\top s) + \frac{1}{2} s^\top s$$

where $\Xi$ is the memory matrix, $s$ is the current state, and $\text{lse}$ is the log-sum-exp function. Update steps follow the gradient of this energy via a softmax-weighted sum over memories.

### Glutamate Dynamics (Stage 1)

$$\frac{dg}{dt} = \alpha \cdot \overline{|s|} - \gamma \cdot a_{\text{astro}} \cdot g$$

As astrocyte function $a_{\text{astro}}$ declines, glutamate clearance weakens and `g` evolves based on neural activity. `g` modulates both the effective inverse temperature ($\beta_{\text{eff}} = \beta \cdot g$) and the update step size ($\eta_{\text{eff}} = \eta \cdot g$).

### Synaptic Loss (Stage 2)

Rather than a global weight scalar, synapses are dropped per-weight using a binary mask sampled at rate `target_scale`. This structurally degrades the memory matrix and cannot be recovered by softmax normalisation.

### Neuron Deletion (Stage 3)

Neurons are silenced stochastically. Survival probability is biased by activity level — more active neurons are more likely to be deleted, consistent with excitotoxicity models of neurodegeneration.

---

## Experiments

### 1. Stage Progression
Single-flag retrieval (Malaysia) run through each stage cumulatively. Energy and retrieved state visualised at each stage.

### 2. Noise Robustness
Corruption fraction swept from 0.1 → 0.9. Measures how much noise the network can tolerate before retrieval fails, across all four conditions (healthy + 3 stages).

### 3. Memory Capacity
Number of stored patterns swept from 5 → 50. Measures retrieval quality as interference increases, across all four conditions.

### 4. Similarity Trajectories
Per-step similarity to each stored pattern plotted over time, showing whether the network converges to the correct memory, a distractor, or fails entirely.

---

## Requirements

```
numpy
matplotlib
pandas
pickle
wget
```

The `hopfield_networks_utils.py` dependency is downloaded automatically from [TomGeorge1234/HopfieldNetworkTutorial](https://github.com/TomGeorge1234/HopfieldNetworkTutorial) if not present.

---

## Acknowledgements

- Hopfield Network utilities by [Tom George](https://github.com/TomGeorge1234)
- Flag dataset by [nmohapatra](https://www.kaggle.com/nmohapatra) on Kaggle
- Modern Hopfield Network theory: Ramsauer et al. (2020), *Hopfield Networks is All You Need*
