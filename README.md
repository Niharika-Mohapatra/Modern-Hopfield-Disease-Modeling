# Modelling Neurodegeneration with Modern Hopfield Networks

## Overview

This project uses a **Modern Hopfield Network** to simulate the progressive memory failure seen in neurodegenerative diseases such as Alzheimer's. Greyscale images of Asian country flags serve as memory patterns. The network's ability to retrieve a corrupted flag represents a brain's ability to recall a degraded memory.

Three biologically-motivated stages of decline are modelled sequentially:

| Stage | Biological Basis | Model Mechanism |
|-------|-----------------|-----------------|
| **Stage 1** | Astrocyte dysfunction | Glutamate neuromodulation (`g`) dysregulates, altering retrieval sharpness and step size |
| **Stage 2** | Synaptic pruning | Memory patterns are structurally corrupted via per-synapse dropout |
| **Stage 3** | Neuronal death | Neurons are randomly silenced; active neurons face higher deletion probability (excitotoxicity) |
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

## Results

### Noise Robustness

The noise robustness test sweeps input corruption from 10% to 90% and measures final similarity to the target flag across all four conditions. In the healthy condition, retrieval remains reliable up to approximately 50% corruption, after which similarity degrades sharply. Stage 1 largely preserves this robustness curve, again confirming that glutamate dysregulation alone does not significantly impair the network's error-correction capacity. Stages 2 and 3 shift the curve downward and leftward — the network begins failing at lower noise levels, and the floor similarity under high noise is reduced. By Stages 1–2–3, meaningful retrieval (similarity > 0.5) is only achievable under very low noise conditions (~10–20% corruption), consistent with a brain that can only recall memories when given a near-complete cue.

![alt text](https://github.com/Niharika-Mohapatra/Modern-Hopfield-Disease-Modeling/blob/174f746fba71298fd3079d64404fe0b4d1956c21/Noise_Robustness_results.png)

### Memory Capacity

Retrieval similarity is measured as the number of stored patterns increases from 5 to 50, averaged over 3 random distractor sets. In the healthy condition, similarity remains relatively stable up to ~30 stored patterns before declining, consistent with the theoretical capacity scaling of Modern Hopfield Networks. Stage 1 again shows minimal deviation from healthy. Stages 2 and 3 compress the effective capacity window — retrieval begins degrading at lower pattern counts, and the rate of decline steepens. This mirrors clinical observations where neurodegenerative patients lose access to a wider range of memories as the disease progresses, not just the most recently formed ones.

![alt text](https://github.com/Niharika-Mohapatra/Modern-Hopfield-Disease-Modeling/blob/2ee50c0c4d65bc03e2ae923484151874af47e216/Memory_Capacity_results.png)

## Requirements

```
numpy
matplotlib
pandas
pickle
wget
```

The `hopfield_networks_utils.py` dependency is downloaded automatically from [TomGeorge1234/HopfieldNetworkTutorial](https://github.com/TomGeorge1234/HopfieldNetworkTutorial).

---

## Acknowledgements

- Hopfield Network utilities by [Tom George](https://github.com/TomGeorge1234)
- Modern Hopfield Network theory: Ramsauer et al. (2020), *Hopfield Networks is All You Need*
