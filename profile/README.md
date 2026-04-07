# OmniVoice: Detailed Mathematical Architecture

---

## Table of Contents

- [OmniVoice: Detailed Mathematical Architecture](#omnivoice-detailed-mathematical-architecture)
  - [Table of Contents](#table-of-contents)
  - [1. System Overview](#1-system-overview)
  - [2. Audio Tokenizer (HiggsAudio-V2 RVQ)](#2-audio-tokenizer-higgsaudio-v2-rvq)
  - [3. Sequence Representation](#3-sequence-representation)
    - [3.1 Style Tokens $S$](#31-style-tokens-s)
    - [3.2 Text Tokens $W$](#32-text-tokens-w)
    - [3.3 Audio Mask](#33-audio-mask)
  - [4. Model Architecture](#4-model-architecture)
    - [4.1 Multi-Codebook Embedding Layer](#41-multi-codebook-embedding-layer)
    - [4.2 Transformer LLM Backbone](#42-transformer-llm-backbone)
    - [4.3 Multi-Codebook Audio Head](#43-multi-codebook-audio-head)
  - [5. Training: Masked Token Prediction](#5-training-masked-token-prediction)
    - [5.1 Masking Strategy](#51-masking-strategy)
    - [5.2 Loss Function](#52-loss-function)
    - [5.3 Conditioning Dropout (CFG Training)](#53-conditioning-dropout-cfg-training)
  - [6. Inference: Iterative Unmasking with CFG](#6-inference-iterative-unmasking-with-cfg)
    - [6.1 Timestep Schedule](#61-timestep-schedule)
    - [6.2 Unmasking Schedule](#62-unmasking-schedule)
    - [6.3 Classifier-Free Guidance](#63-classifier-free-guidance)
    - [6.4 Token Prediction \& Scoring](#64-token-prediction--scoring)
    - [6.5 Position Selection](#65-position-selection)
    - [6.6 Layer Priority Penalty](#66-layer-priority-penalty)
    - [6.7 Full Iterative Decoding Algorithm](#67-full-iterative-decoding-algorithm)
  - [7. Duration Estimation](#7-duration-estimation)
  - [8. Voice Cloning \& Voice Design Conditioning](#8-voice-cloning--voice-design-conditioning)
    - [8.1 Voice Cloning](#81-voice-cloning)
    - [8.2 Voice Design](#82-voice-design)
    - [8.3 Auto Voice](#83-auto-voice)
  - [9. Long-Form Generation via Chunking](#9-long-form-generation-via-chunking)
  - [10. Evaluation Metrics](#10-evaluation-metrics)
    - [10.1 Word Error Rate (WER)](#101-word-error-rate-wer)
    - [10.2 Speaker Similarity (SIM)](#102-speaker-similarity-sim)
    - [10.3 Mean Opinion Score (MOS) — UTMOS](#103-mean-opinion-score-mos--utmos)
  - [11. Hyperparameter Reference Table](#11-hyperparameter-reference-table)
    - [Architecture](#architecture)
    - [Training](#training)
    - [Inference](#inference)
  - [Summary Diagram](#summary-diagram)

---

## 1. System Overview

OmniVoice is a **zero-shot, omnilingual TTS system** that factorises speech synthesis into two stages:

$$
\underbrace{x_{\text{wav}}}_{\text{raw waveform}} \xrightarrow{\mathcal{E}} \underbrace{Z \in \mathbb{Z}^{C \times T}}_{\text{discrete audio codes}} \xleftarrow{\mathcal{D}} \underbrace{f_\theta(s, w, Z_{\text{prompt}})}_{\text{Diffusion LM}}
$$

| Symbol                               | Meaning                                                  |
| ------------------------------------ | -------------------------------------------------------- |
| $\mathcal{E}$                        | RVQ audio encoder (HiggsAudio-V2)                        |
| $\mathcal{D}$                        | RVQ audio decoder                                        |
| $Z \in \{0,\dots,V-1\}^{C \times T}$ | $C=8$ codebooks, $T$ frames at $f_r$ fps                 |
| $s$                                  | Style tokens: language + speaker instruction             |
| $w$                                  | Text tokens (transcript + optional reference transcript) |
| $Z_{\text{prompt}}$                  | Optional reference audio tokens (voice cloning)          |
| $V = 1024$                           | Codebook vocabulary size; $V+1 = 1025$ with mask token   |

**Key design choice**: Rather than predicting audio codes autoregressively (left-to-right), OmniVoice employs a **masked discrete diffusion** formulation where all target positions are initially masked and unmasked iteratively over $K$ steps, analogous to masked diffusion language models (MDLMs) and MaskGIT.

---

## 2. Audio Tokenizer (HiggsAudio-V2 RVQ)

The audio waveform $x \in \mathbb{R}^{1 \times L_{\text{wav}}}$ at $f_s = 24{,}000$ Hz is compressed by a Residual Vector Quantizer:

$$
Z = \mathcal{E}(x) = \text{RVQ}_C(e(x))
$$

where $e(\cdot)$ is a convolutional encoder and $\text{RVQ}_C$ applies $C = 8$ successive VQ layers. Each layer $c$ encodes the residual from prior layers:

$$
\mathbf{r}^{(0)} = e(x), \quad z^{(c)} = \arg\min_{v \in \mathcal{B}^{(c)}} \|\mathbf{r}^{(c-1)} - \mathbf{e}_v^{(c)}\|_2^2, \quad \mathbf{r}^{(c)} = \mathbf{r}^{(c-1)} - \mathbf{e}_{z^{(c)}}^{(c)}
$$

- **Frame rate**: $f_r$ frames/sec (e.g., $\approx 75$ fps for 24kHz with hop length $H = 320$), so $T = \lfloor L_{\text{wav}} / H \rfloor$
- **Codebook size**: $|\mathcal{B}^{(c)}| = V = 1024$ for each $c$
- **Special token**: $z_{\text{mask}} = 1024$ (the $V\!+\!1$-th entry), used as the diffusion mask

**Decoding** reconstructs the waveform from all 8 code streams:

$$
\hat{x} = \mathcal{D}\!\left(\sum_{c=1}^{C} \mathbf{e}_{z^{(c)}}^{(c)}\right)
$$

---

## 3. Sequence Representation

At both training and inference, a single sample is encoded into a **flat token sequence** of shape $[C, L]$ where $L = L_s + L_w + L_p + L_t$:

$$
\mathbf{X} = \big[\underbrace{S}_{L_s \text{ style}}\; \big|\; \underbrace{W}_{L_w \text{ text}}\; \big|\; \underbrace{Z_{\text{prompt}}}_{L_p \text{ ref audio}}\; \big|\; \underbrace{Z_{\text{target}}}_{L_t \text{ target audio}}\big]
$$

### 3.1 Style Tokens $S$

Style tokens are formed by the special-token template (as plain text that is BPE-tokenized):

```
[<|denoise|>] <|lang_start|> {lang_id} <|lang_end|> <|instruct_start|> {instruct} <|instruct_end|>
```

- `<|denoise|>` appears only when a reference audio prompt is present (voice cloning with denoising)
- `lang_id` ∈ `{en, zh, ja, ..., None}` — 600+ language codes or `None` for language-agnostic mode
- `instruct` ∈ `{male, female, child, elderly, low pitch, british accent, ...}` or `None`

Style tokens carry **no loss** during training: $\mathbf{Y}_s = -100$ (ignore index).

### 3.2 Text Tokens $W$

$$
W = \texttt{Tokenizer}\!\left(\texttt{<|text\_start|>}\; [\text{ref\_text} + \text{" "}]\; \text{target\_text}\; \texttt{<|text\_end|>}\right)
$$

Text tokens also carry **no loss**: $\mathbf{Y}_w = -100$.

### 3.3 Audio Mask

A boolean mask $\mathbf{m} \in \{0,1\}^L$ marks which positions hold audio tokens:

$$
m_l = \begin{cases} 1 & \text{if } l \geq L_s + L_w \quad \text{(audio region)} \\ 0 & \text{otherwise} \end{cases}
$$

---

## 4. Model Architecture

### 4.1 Multi-Codebook Embedding Layer

For each sequence position $l$, the embedding is determined by whether $l$ is in the audio region:

$$
\mathbf{h}_l^{(0)} = \begin{cases}
\displaystyle\sum_{c=1}^{C} \mathbf{E}_{\text{audio}}\!\left[x_l^{(c)} + (c-1) \cdot V\right] & \text{if } m_l = 1 \\[6pt]
\mathbf{E}_{\text{text}}\!\left[x_l^{(1)}\right] & \text{if } m_l = 0
\end{cases}
$$

where:

- $\mathbf{E}_{\text{audio}} \in \mathbb{R}^{(C \cdot V_+) \times d}$ is a shared embedding table for all codebooks, with $V_+ = V + 1 = 1025$ (includes mask token)
- The codebook offset $(c-1) \cdot V_+$ ensures layer-$c$ tokens address a distinct shard of the embedding table
- $\mathbf{E}_{\text{text}} \in \mathbb{R}^{|\mathcal{V}_{\text{text}}| \times d}$ is the LLM's standard text embedding
- $d$ = LLM hidden dimension (e.g., $d = 1024$ for Qwen3-0.6B)

In matrix form for a full sequence:

$$
\mathbf{H}^{(0)} = \text{where}\!\left(\mathbf{m}^{\uparrow d},\;\; \sum_{c=1}^{C} \mathbf{E}_{\text{audio}}\!\left[\mathbf{X}^{(c)} + (c-1)\cdot V_+\right],\;\; \mathbf{E}_{\text{text}}\!\left[\mathbf{X}^{(1)}\right]\right)
$$

where $\mathbf{m}^{\uparrow d}$ broadcasts the mask along the embedding dimension.

### 4.2 Transformer LLM Backbone

OmniVoice wraps an off-the-shelf causal LLM (default: **Qwen3-0.6B**, $N = 28$ transformer layers, $d = 1024$, 16 attention heads with GQA). The backbone is used as a **full-sequence (non-causal) encoder** during training via document-aware packed attention.

**Packed Sequence Attention Mask** (FlexAttention):

For a batch with multiple packed documents, the block mask restricts attention to same-document tokens:

$$
A_{q,k} = \mathbf{1}\!\left[\text{doc}(q) = \text{doc}(k)\right]
$$

This eliminates cross-contamination between samples packed into the same tensor without requiring padding.

**Each Transformer Layer** $i \in \{1, \dots, N\}$:

$$
\mathbf{H}^{(i)} = \text{TransformerLayer}_i\!\left(\mathbf{H}^{(i-1)},\; A\right)
$$

comprising:

- **RMSNorm** pre-normalization
- **Multi-head attention** (with RoPE positional encoding, GQA)
- **SwiGLU FFN**
- Residual connections throughout

**Position IDs**: Assigned per-document within each packed sequence, so each document's tokens have positions $0, 1, 2, \dots$

### 4.3 Multi-Codebook Audio Head

After the final layer, a single linear projection maps to all $C$ codebook logits simultaneously:

$$
\mathbf{O} = \mathbf{H}^{(N)} \mathbf{W}_{\text{head}}^\top \in \mathbb{R}^{B \times L \times (C \cdot V_+)}
$$

Reshaped to per-codebook logits:

$$
\ell^{(c)}_l = \mathbf{O}_l\!\left[(c-1) V_+ : c V_+\right] \in \mathbb{R}^{V_+}, \quad c = 1, \dots, C
$$

Stacked: $\boldsymbol{\ell} \in \mathbb{R}^{B \times C \times L \times V_+}$

The weight matrix $\mathbf{W}_{\text{head}} \in \mathbb{R}^{(C \cdot V_+) \times d}$ is **not tied** to the audio embedding table $\mathbf{E}_{\text{audio}}$.

---

## 5. Training: Masked Token Prediction

### 5.1 Masking Strategy

For each training sample, the audio token sequence $Z \in \{0,\dots,V-1\}^{C \times T}$ is split into a **prompt region** $[0, L_p)$ and a **generation region** $[L_p, T)$.

**Prompt length** is sampled uniformly:

$$
L_p = \lfloor \rho_p \cdot T \rfloor, \quad \rho_p \sim \mathcal{U}[\rho_{\min}, \rho_{\max}] = \mathcal{U}[0.0, 0.3]
$$

**Mask ratio** over the generation region:

$$
r_m \sim \mathcal{U}[\rho_{m,\min}, \rho_{m,\max}] = \mathcal{U}[0.0, 1.0]
$$

A binary mask $M \in \{0,1\}^{C \times (T - L_p)}$ is drawn independently per token:

$$
M_{c,t} \sim \text{Bernoulli}(r_m), \quad c \in [C],\; t \in [T - L_p]
$$

**Masked input** fed to the model:

$$
\tilde{Z}_{c,t} = \begin{cases}
z_{\text{mask}} & \text{if } t \geq L_p \text{ and } M_{c,t-L_p} = 1 \\
Z_{c,t} & \text{otherwise}
\end{cases}
$$

**Labels** (only masked positions incur loss):

$$
Y_{c,t} = \begin{cases}
-100 & \text{if } t < L_p \quad \text{(no loss on prompt)} \\
-100 & \text{if } t \geq L_p \text{ and } M_{c,t-L_p} = 0 \quad \text{(unmasked — already known)} \\
Z_{c,t} & \text{if } t \geq L_p \text{ and } M_{c,t-L_p} = 1 \quad \text{(predict this)}
\end{cases}
$$

### 5.2 Loss Function

The loss is a **codebook-weighted cross-entropy** over all masked positions:

$$
\mathcal{L}_c = \frac{\displaystyle\sum_{b,t} \mathbf{1}[Y^{(b)}_{c,t} \neq -100]\cdot \text{CE}\!\left(\boldsymbol{\ell}^{(c,b)}_t,\; Y^{(b)}_{c,t}\right)}{\displaystyle\sum_{b,t} \mathbf{1}[Y^{(b)}_{c,t} \neq -100]}
$$

$$
\mathcal{L} = \sum_{c=1}^{C} \tilde{w}_c \cdot \mathcal{L}_c
$$

where the normalized codebook weights are:

$$
\tilde{w}_c = \frac{w_c}{\sum_{c'} w_{c'}}, \quad (w_1,\dots,w_8) = (8,8,6,6,4,4,2,2)
$$

**Rationale**: Lower codebook layers (coarse structure, $w_c$ high) are penalized more heavily than higher layers (fine acoustic detail, $w_c$ low), which matches the semantic importance hierarchy in RVQ.

### 5.3 Conditioning Dropout (CFG Training)

With probability $p_{\text{drop}} = 0.1$, **all conditioning** is dropped:

$$
\text{if drop\_cond}: \quad \rho_p = 0,\; \text{drop\_text} = \text{True},\; \text{lang} = \text{None},\; \text{instruct} = \text{None}
$$

This trains the model to score tokens unconditionally, enabling classifier-free guidance at inference.

Additional stochastic conditioning:

- Language token included with prob $p_{\text{lang}} = 0.8$
- Instruction token included with prob $p_{\text{inst}} = 1.0$, but voice-design-only mode (no audio prompt) with prob $p_{\text{only\_inst}} = 0.5$
- Pinyin pronunciation used for Chinese text with prob $p_{\text{pinyin}} = 0.3$

---

## 6. Inference: Iterative Unmasking with CFG

### 6.1 Timestep Schedule

OmniVoice uses a **shifted linear timestep schedule** borrowed from flow-matching literature. Given $K$ generation steps, define $K+1$ timesteps:

$$
\bar{t}_k = \frac{k}{K}, \quad k = 0, 1, \dots, K
$$

Apply a **time-shift** with parameter $\tau$ (default $\tau = 0.1$):

$$
t_k = \frac{\tau \cdot \bar{t}_k}{1 + (\tau - 1)\cdot \bar{t}_k}
$$

This compresses timesteps toward $t=0$ (low-noise regime), concentrating more steps on the hard early unmasking phase. For $\tau < 1$, the schedule front-loads unmasking:

$$
\frac{dt_k}{d\bar{t}_k}\bigg|_{\bar{t}=0} = \tau, \quad \frac{dt_k}{d\bar{t}_k}\bigg|_{\bar{t}=1} = \frac{\tau}{(1 + (\tau-1))^2} = \tau
$$

### 6.2 Unmasking Schedule

At step $k$, unmask $n_k$ tokens out of the remaining $R_k$ masked tokens. The total token budget is:

$$
N_{\text{total}} = T \cdot C
$$

The number of tokens to unmask at step $k$:

$$
n_k = \begin{cases}
R_k & \text{if } k = K-1 \quad \text{(last step: unmask all remaining)} \\
\min\!\left(\left\lceil N_{\text{total}} \cdot (t_{k+1} - t_k) \right\rceil,\; R_k\right) & \text{otherwise}
\end{cases}
$$

$$
R_0 = N_{\text{total}}, \quad R_{k+1} = R_k - n_k
$$

### 6.3 Classifier-Free Guidance

At each step, the model runs **two forward passes** on a doubled batch:

1. **Conditional** ($\text{cond}$): Full input sequence with style + text + reference audio prompt
2. **Unconditional** ($\text{uncond}$): Only the target audio tokens (no text/style/prompt conditioning)

The guided log-probabilities combine both:

$$
\log p_\phi^{\text{guided}}(z) = \log\mathrm{softmax}\!\left(\log p_\phi^{\text{cond}}(z) + \lambda \cdot \left[\log p_\phi^{\text{cond}}(z) - \log p_\phi^{\text{uncond}}(z)\right]\right)
$$

where $\lambda = 2.0$ is the guidance scale. Expanding:

$$
\log p_\phi^{\text{guided}}(z) = \log\mathrm{softmax}\!\left((1+\lambda)\,\log p_\phi^{\text{cond}}(z) - \lambda\,\log p_\phi^{\text{uncond}}(z)\right)
$$

When $\lambda = 0$, this reduces to the conditional distribution. The logit-space formulation preserves numerical stability by operating on log-softmax outputs before re-normalizing.

The mask token $z_{\text{mask}}$ is excluded from valid predictions:

$$
\log p_\phi^{\text{guided}}(z_{\text{mask}}) = -\infty
$$

### 6.4 Token Prediction & Scoring

**Greedy prediction** (default, `class_temperature = 0`):

$$
\hat{z}_{c,t} = \arg\max_{v \neq z_{\text{mask}}} \log p_\phi^{\text{guided}}(v \mid c, t)
$$

**Stochastic prediction** (`class_temperature > 0`): Apply top-$k$ filtering (retain top 10% of vocabulary) then Gumbel-argmax sampling:

$$
\tilde{\ell}_{c,t,v} = \ell_{c,t,v} + g_v, \quad g_v \overset{\text{iid}}{\sim} \text{Gumbel}(0,1)
$$

$$
\hat{z}_{c,t} = \arg\max_{v \in \text{Top-}k} \frac{\tilde{\ell}_{c,t,v}}{\tau_{\text{cls}}}
$$

**Confidence score** for position selection:

$$
s_{c,t} = \max_v \log p_\phi^{\text{guided}}(v \mid c, t) = \log p_\phi^{\text{guided}}(\hat{z}_{c,t} \mid c, t)
$$

### 6.5 Position Selection

From all still-masked positions $\mathcal{M}_k = \{(c,t) : \tilde{Z}_{c,t}^{(k)} = z_{\text{mask}}\}$, select the $n_k$ positions to unmask at step $k$.

**Gumbel-temperature position sampling** (`position_temperature > 0`):

$$
\tilde{s}_{c,t} = \frac{s_{c,t}}{\tau_{\text{pos}}} + g_{c,t}, \quad g_{c,t} \overset{\text{iid}}{\sim} \text{Gumbel}(0,1)
$$

**Layer priority penalty** (penalizes selecting higher-index codebook layers first):

$$
\hat{s}_{c,t} = \tilde{s}_{c,t} - \beta \cdot (c - 1)
$$

where $\beta = 5.0$ (layer penalty factor). This encourages coarser codebook layers ($c=1$) to be unmasked before finer ones ($c=C$), which mirrors the hierarchical nature of RVQ.

**Top-$n_k$ selection** of positions in $\mathcal{M}_k$:

$$
\mathcal{U}_k = \text{Top-}n_k\!\left\{\hat{s}_{c,t} : (c,t) \in \mathcal{M}_k\right\}
$$

**Update**:

$$
Z_{c,t}^{(k+1)} = \begin{cases} \hat{z}_{c,t} & \text{if } (c,t) \in \mathcal{U}_k \\ Z_{c,t}^{(k)} & \text{otherwise} \end{cases}
$$

### 6.6 Layer Priority Penalty

The penalty $\beta \cdot (c-1)$ is subtracted from the confidence score of each token in codebook layer $c$. For 8 layers with $\beta = 5$:

| Layer $c$  | Penalty $\beta(c-1)$ |
| ---------- | -------------------- |
| 1 (coarse) | 0.0                  |
| 2          | 5.0                  |
| 3          | 10.0                 |
| 4          | 15.0                 |
| 5          | 20.0                 |
| 6          | 25.0                 |
| 7          | 30.0                 |
| 8 (fine)   | 35.0                 |

Since log-probabilities are in $(-\infty, 0]$, this penalty ensures that all layer-1 tokens will be preferentially unmasked before layer-2 tokens unless a layer-1 token has extremely low confidence ($< -5$).

### 6.7 Full Iterative Decoding Algorithm

```
Input:  text w, style s, ref audio prompt Z_prompt (optional)
        K=32 steps, λ=2.0 (guidance), τ=0.1 (time shift)
        τ_pos=5.0 (position temp), β=5.0 (layer penalty)

Initialize:  Z^(0) = [z_mask]^{C×T}  (all masked)
             Compute timesteps {t_0,...,t_K} via shifted schedule

For k = 0, 1, ..., K-1:
  # Build cond input:  [s | w | Z_prompt | Z^(k)]
  # Build uncond input: [Z^(k)]

  Run forward pass (doubled batch → 2 forward passes in 1 call):
    ℓ^cond, ℓ^uncond = f_θ(cond_input), f_θ(uncond_input)

  Compute guided log-probs:
    log p^guided = log_softmax(log_softmax(ℓ^cond) + λ·[log_softmax(ℓ^cond) - log_softmax(ℓ^uncond)])

  Predict tokens:    ẑ = argmax log p^guided  (on non-mask vocab)
  Compute scores:    s_{c,t} = log p^guided(ẑ_{c,t})

  Apply Gumbel noise: s̃_{c,t} = s_{c,t}/τ_pos + Gumbel(0,1)
  Apply layer penalty: ŝ_{c,t} = s̃_{c,t} - β·(c-1)

  Mask out already-unmasked:  ŝ_{c,t} = -∞  if Z^(k)_{c,t} ≠ z_mask

  Compute n_k from schedule
  Select top-n_k positions by ŝ → U_k

  Z^(k+1)_{c,t} = ẑ_{c,t}  if (c,t) ∈ U_k,  else Z^(k)_{c,t}

Output:  Z^(K)  →  x̂ = D(Z^(K))
```

---

## 7. Duration Estimation

When no explicit `duration` is given, OmniVoice estimates the target frame count $T$ using a **rule-based phonetic weight model** that covers 600+ languages by Unicode script block.

**Character weight function** $w : \Sigma \to \mathbb{R}_{> 0}$:

| Script family           | $w(c)$ | Examples                    |
| ----------------------- | ------ | --------------------------- |
| CJK logographic         | 3.0    | Chinese, Japanese Kanji     |
| Hangul syllabic         | 2.5    | Korean                      |
| Kana syllabic           | 2.2    | Japanese Hiragana/Katakana  |
| Ethiopic                | 3.0    | Amharic                     |
| Indic abugida           | 1.8    | Hindi, Bengali, Tamil       |
| Thai/Lao                | 1.5    | Thai, Lao                   |
| Arabic/Hebrew abjad     | 1.5    | Arabic, Persian, Hebrew     |
| Latin/Cyrillic alphabet | 1.0    | English, Russian (baseline) |
| Digit                   | 3.5    | Number expansion            |
| Punctuation             | 0.5    | Pause markers               |
| Space                   | 0.2    | Word boundary               |
| Diacritic mark          | 0.0    | Silent modifiers            |

**Total phonetic weight** of a string $u$:

$$
W(u) = \sum_{i=1}^{|u|} w(u_i)
$$

**Speed estimation** from reference audio:

$$
v_{\text{speak}} = \frac{W(u_{\text{ref}})}{d_{\text{ref}}} \quad \text{[weight units per frame]}
$$

**Duration estimate** for target text $u_{\text{tgt}}$:

$$
\hat{d}_{\text{tgt}} = \frac{W(u_{\text{tgt}})}{v_{\text{speak}}}
$$

**Short-duration boost** (power-law correction for short utterances with threshold $d_0 = 50$ frames):

$$
d_{\text{final}} = \begin{cases}
d_0 \cdot \left(\dfrac{\hat{d}_{\text{tgt}}}{d_0}\right)^{1/\gamma} & \text{if } \hat{d}_{\text{tgt}} < d_0 \\[8pt]
\hat{d}_{\text{tgt}} & \text{otherwise}
\end{cases}
$$

where $\gamma = 3.0$ (boost strength). This prevents under-estimation for very short sentences.

**Speed control**: Applying a speed factor $\alpha > 0$:

$$
T = \max\!\left(1,\; \left\lfloor d_{\text{final}} / \alpha \right\rfloor\right)
$$

---

## 8. Voice Cloning & Voice Design Conditioning

### 8.1 Voice Cloning

Given a reference waveform $x_{\text{ref}}$:

1. **Preprocess**: Remove silence, trim to $\leq 20$s, normalize RMS if $\text{RMS}(x_{\text{ref}}) < 0.1$
2. **Tokenize**: $Z_{\text{prompt}} = \mathcal{E}(x_{\text{ref}}) \in \{0,\dots,V-1\}^{C \times L_p}$
3. **Transcribe** (if `ref_text` not given): $w_{\text{ref}} = \text{Whisper}(x_{\text{ref}})$
4. **Build sequence**: $[s \mid w_{\text{ref}} + w_{\text{tgt}} \mid Z_{\text{prompt}} \mid \tilde{Z}_{\text{target}}]$

The RMS ratio is preserved at output:

$$
\hat{x} \leftarrow \hat{x} \cdot \frac{\text{RMS}_{\text{ref}}}{0.1} \quad \text{if RMS}_{\text{ref}} < 0.1
$$

### 8.2 Voice Design

No reference audio. The `instruct` string encodes speaker attributes:

$$
s = \texttt{<|instruct\_start|>}\; \text{instruct} \; \texttt{<|instruct\_end|>}
$$

Supported attribute categories (each mutually exclusive within category):

- **Gender**: `male`, `female`
- **Age**: `child`, `teenager`, `young adult`, `middle-aged`, `elderly`
- **Pitch**: `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch`
- **Style**: `whisper`
- **English accent**: `american`, `british`, `australian`, `indian`, `canadian`, `scottish`, ...
- **Chinese dialect**: `四川话`, `陕西话`, `东北话`, `粤语`, ...

Output is **peak-normalized** to 0.5 when no RMS reference is available.

### 8.3 Auto Voice

Both prompt and instruct are `None`. The model operates purely on language and text tokens, sampling a voice from the learned prior. The unconditional branch in CFG effectively regularizes toward natural-sounding voices.

---

## 9. Long-Form Generation via Chunking

For utterances whose estimated frame count exceeds $T_{\text{thresh}} = 30\text{s} \times f_r$ frames, the text is chunked at punctuation boundaries:

$$
u = [u_1, u_2, \dots, u_P], \quad |u_i| \approx \frac{f_{\text{chunk}} \cdot f_r}{\bar{\alpha}}
$$

where $f_{\text{chunk}} = 15$s and $\bar{\alpha} = T_{\text{tgt}} / |u|$ is the average tokens per character.

**Voice cloning mode**: Each chunk $u_i$ is generated independently using the original $Z_{\text{prompt}}$.

**Auto/design mode**: Chunk $u_1$ is generated freely; subsequent chunks $u_i, i > 1$ use chunk $u_{i-1}$'s output tokens as $Z_{\text{prompt}}$, ensuring voice consistency across chunks.

**Cross-fade joining**: Decoded chunk waveforms $\hat{x}_1, \dots, \hat{x}_P$ are concatenated with a cross-fade:

$$
\hat{x} = \bigoplus_{\text{xfade}} (\hat{x}_1, \hat{x}_2, \dots, \hat{x}_P)
$$

---

## 10. Evaluation Metrics

### 10.1 Word Error Rate (WER)

Speech intelligibility is measured by ASR-WER:

$$
\text{WER} = \frac{S + D + I}{N} \times 100\%
$$

where $S, D, I, N$ are substitutions, deletions, insertions, and total reference words respectively. Multiple ASR backends are supported: Whisper, HuBERT, SenseVoice, MiniMax.

### 10.2 Speaker Similarity (SIM)

Speaker similarity uses an **ECAPA-TDNN + WavLM** speaker encoder:

$$
\text{SIM}(\hat{x}, x_{\text{ref}}) = \cos\!\left(\mathbf{e}(\hat{x}),\; \mathbf{e}(x_{\text{ref}})\right) = \frac{\mathbf{e}(\hat{x})^\top \mathbf{e}(x_{\text{ref}})}{\|\mathbf{e}(\hat{x})\| \cdot \|\mathbf{e}(x_{\text{ref}})\|}
$$

The speaker encoder $\mathbf{e}(\cdot)$ maps a waveform to an $\mathbb{R}^{192}$ d-vector via:

1. **WavLM Large** (24-layer transformer, 1024-d) — outputs 25 layer representations
2. **Learnable weighted sum** over layers: $\tilde{\mathbf{f}} = \sum_{l=0}^{24} \text{softmax}(\boldsymbol{\alpha})_l \cdot \mathbf{F}_l$
3. **ECAPA-TDNN head**: Conv1d → SE-Res2Blocks → attentive statistics pooling → 192-d embedding

### 10.3 Mean Opinion Score (MOS) — UTMOS

**UTMOS** (Automatic MOS Predictor) estimates naturalness via a fine-tuned SSL model. The predicted score $\hat{\mu} \in [1, 5]$ approximates human perceptual ratings.

---

## 11. Hyperparameter Reference Table

### Architecture

| Parameter                | Symbol            | Value      |
| ------------------------ | ----------------- | ---------- |
| LLM backbone             | —                 | Qwen3-0.6B |
| LLM hidden dim           | $d$               | 1024       |
| LLM layers               | $N$               | 28         |
| Number of codebooks      | $C$               | 8          |
| Codebook vocabulary size | $V$               | 1024       |
| Mask token ID            | $z_{\text{mask}}$ | 1024       |
| Audio frame rate         | $f_r$             | ~75 fps    |
| Audio sample rate        | $f_s$             | 24,000 Hz  |
| Hop length (RVQ)         | $H$               | 320        |

### Training

| Parameter               | Symbol                       | Value               |
| ----------------------- | ---------------------------- | ------------------- |
| Learning rate           | $\eta$                       | $1 \times 10^{-4}$  |
| LR schedule             | —                            | Cosine              |
| Warmup ratio            | —                            | 3%                  |
| Weight decay            | $\lambda_{\text{wd}}$        | 0.01                |
| Max gradient norm       | —                            | 1.0                 |
| Batch tokens            | —                            | 8192                |
| Steps                   | —                            | 300,000             |
| Mixed precision         | —                            | BF16                |
| Prompt ratio range      | $[\rho_{\min}, \rho_{\max}]$ | $[0.0, 0.3]$        |
| Mask ratio range        | $[r_{\min}, r_{\max}]$       | $[0.0, 1.0]$        |
| Drop-conditioning ratio | $p_{\text{drop}}$            | 0.10                |
| Language token ratio    | $p_{\text{lang}}$            | 0.80                |
| Instruction token ratio | $p_{\text{inst}}$            | 1.00                |
| Only-instruct ratio     | $p_{\text{only\_inst}}$      | 0.50                |
| Pinyin usage ratio      | $p_{\text{pinyin}}$          | 0.30                |
| Codebook weights        | $(w_1,\ldots,w_8)$           | $(8,8,6,6,4,4,2,2)$ |

### Inference

| Parameter            | Symbol              | Default      |
| -------------------- | ------------------- | ------------ |
| Decoding steps       | $K$                 | 32           |
| Guidance scale       | $\lambda$           | 2.0          |
| Time-shift           | $\tau$              | 0.1          |
| Position temperature | $\tau_{\text{pos}}$ | 5.0          |
| Class temperature    | $\tau_{\text{cls}}$ | 0.0 (greedy) |
| Layer penalty factor | $\beta$             | 5.0          |
| Chunk duration       | $f_{\text{chunk}}$  | 15.0 s       |
| Chunk threshold      | $T_{\text{thresh}}$ | 30.0 s       |

---

## Summary Diagram

```
Input:
  text w  ──────────────────────────────────────────────────────────────┐
  style s (lang + instruct)  ───────────────────────────────────────────┤
  ref audio x_ref  → HiggsAudio-V2 RVQ → Z_prompt  ───────────────────┤
                                                                         ▼
                        ┌─────────────────────────────────────────────────────┐
                        │  Sequence: [s | w_ref + w_tgt | Z_prompt | Z̃_tgt]  │
                        │  Z̃_tgt initialized to all MASK tokens              │
                        └─────────────────────────────────────────────────────┘
                                              │
                        ┌─────────────────────▼──────────────────────────────┐
                        │           Multi-Codebook Embedding                 │
                        │  text pos → E_text[·]; audio pos → Σ_c E_audio[·]  │
                        └─────────────────────┬──────────────────────────────┘
                                              │ H^(0) ∈ R^{B×L×d}
                        ┌─────────────────────▼──────────────────────────────┐
                        │          Transformer LLM (Qwen3-0.6B)              │
                        │  28 layers, RoPE + GQA + SwiGLU, FlexAttention     │
                        └─────────────────────┬──────────────────────────────┘
                                              │ H^(N) ∈ R^{B×L×d}
                        ┌─────────────────────▼──────────────────────────────┐
                        │         Audio Head: Linear(d → C·V_+)              │
                        │  logits ∈ R^{B×C×L×V_+}                            │
                        └─────────────────────┬──────────────────────────────┘
                                              │
                        ┌─────────────────────▼──────────────────────────────┐
                        │    Iterative Unmasking (K=32 steps)                │
                        │  CFG: log p^guided = log p^cond + λ(p^cond-p^unc) │
                        │  Gumbel position sampling + layer penalty           │
                        └─────────────────────┬──────────────────────────────┘
                                              │ Z^(K) ∈ {0..V-1}^{C×T}
                        ┌─────────────────────▼──────────────────────────────┐
                        │      HiggsAudio-V2 Decoder D(Z^(K))                │
                        └─────────────────────┬──────────────────────────────┘
                                              │
                                    x̂_wav @ 24 kHz
```
