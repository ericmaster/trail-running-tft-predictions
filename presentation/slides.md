---
title: Application of Temporal Fusion Transformer to Trail Running Predictions
theme: night
css: assets/custom.css
highlightTheme: monokai
revealOptions:
  transition: slide
  transitionSpeed: fast
  controls: true
  progress: true
  slideNumber: true
  hash: true
  center: true
---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)" -->

# Application of Temporal Fusion Transformer to Trail Running Predictions

### Master's Degree Final Project

**Eric Aguayo**

Universidad San Francisco de Quito

December 2025

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #16213e 0%, #1a1a2e 100%)" -->

## 🏔️ The Challenge

<div style="display: flex; justify-content: space-around; align-items: center; align-items: flex-start;">
<div style="flex: 1;">

### Trail Running Complexity

- Long Endurance activities (3-6 hours)
- Extreme **elevation changes**
- Complex **fatigue dynamics**
- **Nutrition & hydration** critical

</div>
<div style="flex: 1;">

### Why Prediction Matters

- ⚠️ **30% under-prediction** → dehydration, bonking
- 📊 **Over-prediction** → suboptimal pacing
- 🎯 **Goal**: Accurate race time estimation

</div>
</div>

----

<!-- .slide: data-background-gradient="linear-gradient(135deg, #0f3460 0%, #16213e 100%)" -->

### ⚠️ The problem: cold-start predictions

> *"How can we predict race completion time **before the race begins**, without any data from the current session?"*

<div class="fragment">

### Traditional Methods

| Method | Limitation |
|--------|------------|
| Average pace | Ignores terrain complexity |
| Naismith's rule | No fatigue modeling |
| Regression | Misses temporal dependencies |

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%)" -->

## 💡 Our Solution

### Temporal Fusion Transformer (TFT)

<div style="display: flex; justify-content: space-around; align-items: flex-start;">
<div style="flex: 1;">

**Architecture Components:**
<div style="text-align: start; justify-content: flex-start;">
<div class="fragment">
- 🔄 LSTM Encoder-Decoder
</div>
<div class="fragment">
- 🎯 Multi-head Self-Attention
</div>
<div class="fragment">
- 🚪 Gated Residual Networks
</div>
<div class="fragment">
- 📊 Variable Selection Networks
</div>
</div>

----

<!-- .slide: data-background-gradient="linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%)" -->

**Key Advantages:**
- Handles **known** vs **unknown** futures
- Learns **long-range** dependencies
- Provides **interpretable** attention weights
- Supports **multi-target** forecasting

</div>
</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #16213e 0%, #1a1a2e 100%)" -->

## 🔬 Key Contributions

<div style="font-size: 0.9em;">

| # | Contribution | Impact |
|---|--------------|--------|
| 1 | **Novel Application** | First TFT for trail running |
| 2 | **Cold-Start Methodology** | Synthetic encoder approach |
| 3 | **Asymmetric Loss Function** | Corrects under-prediction bias |
| 4 | **Distance-Domain Resampling** | Pace-independent predictions |
| 5 | **Multi-Target Forecasting** | Duration, HR, temp, cadence |
| 6 | **Error Cancellation Analysis** | Robust evaluation guidance |

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #0f3460 0%, #16213e 100%)" -->

## 📊 Data Pipeline

### Distance-Domain Resampling

```
Time Domain (1 sec) → Distance Domain (5 meters)
```

<div style="display: flex; justify-content: space-around;">
<div style="flex: 1;">

**Input Features:**
- Heart Rate
- Altitude
- Cadence
- Speed
- Temperature

</div>
<div style="flex: 1;">

**Derived Features:**
- Elevation diff/gain/loss
- Duration per interval
- Fatigue proxies
- Rolling averages

</div>
</div>

<div class="fragment">

**Dataset:** 106 Polar sessions (79 train / 16 val / 11 test)

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)" -->

## 🧊 Cold-Start Solution

### Synthetic Encoder Approach

$$\mathbf{x}_{synthetic} = \frac{\sum_{s=1}^{S} w_s \cdot \mathbf{x}_{s,0}}{\sum_{s=1}^{S} w_s}$$

<div style="display: flex; justify-content: space-around; align-items: center;">
<div style="flex: 1;">

**How it works:**
1. Weight historical first samples
2. Recent sessions weighted higher
3. Use actual **terrain data** (known)
4. Estimate physiological baseline

</div>
<div style="flex: 1;">

**Why it works:**
- Captures current **fitness level**
- Leverages **population patterns**
- GPS route **preview** available
- No race data needed! 🎯

</div>
</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #16213e 0%, #0f3460 100%)" -->

## ⚖️ Asymmetric Loss Function

### Correcting Under-Prediction Bias

$$\mathcal{L}_{asym} = w \cdot \frac{|y - \hat{y}|}{|y| + |\hat{y}| + \epsilon} \cdot 2$$

<div class="fragment">

$$w = \begin{cases} \alpha & \text{if } y > \hat{y} \text{ (under-prediction)} \\ 1 - \alpha & \text{if } y \leq \hat{y} \text{ (over-prediction)} \end{cases}$$

</div>

<div class="fragment">

**With α = 0.51:** Slight penalty for under-prediction

> ⚠️ *Highly sensitive parameter:* α=0.55 or α=0.60 caused pronounced over-prediction

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%)" -->

## 📈 Results: V1 vs V2

### Cold-Start Inference (24.5 km Session)

| Metric | V1 (SMAPE) | V2 (Asymmetric) | Change |
|--------|------------|-----------------|--------|
| MAE (s/5m) | 1.347 | 1.066 | **-20.9%** |
| Bias (s/5m) | -1.162 | +0.157 | **+113.5%** |
| Actual Duration | 324.9 min | 324.9 min | -- |
| Predicted | 229.9 min | 337.7 min | +46.9% |
| **Accumulated Error** | **-29.2%** | **+3.9%** | ✅ |

<div class="fragment">

**Across all 11 test sessions:** -30.4% → **+3.7%** 🎯

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)" -->

## 📊 Visual Results

### Accumulated Duration Prediction

![V1 vs V2 Comparison](assets/v1_v2_comparison.png)

<small>V2 (asymmetric SMAPE) shows substantially reduced under-prediction bias throughout the entire session</small>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #16213e 0%, #0f3460 100%)" -->

## 🔄 V3: Transfer Learning

### Fine-tuning with Garmin + Nutrition Data

<div style="display: flex; justify-content: space-around;">
<div style="flex: 1;">

**New Features (V3):**
- Rate of Perceived Exertion (RPE)
- Water intake
- Electrolyte intake
- Food intake

*Via NutritionLogger app*

</div>
<div style="flex: 1;">

**Key Finding:**

| Metric | V3 (Garmin) | V2 (Polar) |
|--------|-------------|------------|
| MAE (s/5m) | 0.168 | 0.633 |
| Final Error | +2.2% | +1.8% |

</div>
</div>

<div class="fragment">

> ⚠️ **Error Cancellation Warning:** Lower cumulative error can mask higher per-step error!

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #0f3460 0%, #16213e 100%)" -->

## ⚠️ Limitations & Honest Assessment

<div style="font-size: 0.85em;">

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Single-Athlete Dataset** | Unknown generalization | Multi-athlete validation needed |
| **106 Sessions** | Overfitting risk | Regularization (dropout=0.25) |
| **Geographic Specificity** | Andes-trained only | Test on varied terrain |
| **Missing Features** | Weather, sleep, HRV absent | Future sensor integration |
| **Computational Cost** | GPU required | Model compression needed |

</div>

<div class="fragment">

**V3 Lesson:** 5 sessions insufficient for sparse feature learning → **20+ sessions** recommended

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%)" -->

## 🚀 Practical Applications

<div style="display: flex; justify-content: space-around; flex-wrap: wrap;">

<div style="width: 45%; margin: 10px; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;">

### 🏃 Race Planning
Estimate finish time **before race start** based on route profile

</div>

<div style="width: 45%; margin: 10px; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;">

### ⏱️ Pacing Strategy
Real-time predictions for **pacing adjustments**

</div>

<div style="width: 45%; margin: 10px; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;">

### 🍎 Nutrition Planning
Predicted duration informs **caloric & fluid** needs

</div>

<div style="width: 45%; margin: 10px; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;">

### 📈 Training Optimization
Analyze **predicted vs actual** performance

</div>

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #16213e 0%, #1a1a2e 100%)" -->

## 🔮 Future Work

<div style="font-size: 0.9em;">

1. **Multi-Athlete Datasets** - Platforms like Strava/Garmin Connect

2. **Weather Integration** - Temperature, humidity, wind conditions

3. **Uncertainty Quantification** - Confidence intervals for predictions

4. **Model Compression** - On-device inference for sports watches

5. **Multi-Metric Evaluation** - Both per-step AND cumulative metrics

6. **Session Type Classification** - Training vs. race differentiation

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #0f3460 0%, #16213e 100%)" -->

## ✅ Conclusions

<div style="font-size: 0.9em;">

1. **TFT Successfully Applied** to trail running prediction - *first documented implementation*

2. **Cold-Start Prediction Achieved** via synthetic encoder approach

3. **Asymmetric Loss (α=0.51)** reduced error from **-30.4% to +3.7%**

4. **Distance-Domain Processing** enables pace-independent predictions

5. **Transfer Learning Works** - V2 model transferred to Garmin with +4.1% error

6. **Critical Insight:** Evaluate both **per-step** and **cumulative** metrics

</div>

<div class="fragment" style="margin-top: 30px; font-size: 1.2em;">

> 🎯 *"Conservative over-prediction is preferable for race planning"*

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)" -->

# Thank You!

### Questions?

<div style="margin-top: 50px;">

**Eric Aguayo**

📧 ericmaster@nimblersoft.com

🔗 github.com/ericmaster/tft-predictor

</div>

<small style="margin-top: 50px; display: block;">Universidad San Francisco de Quito | Maestría en Inteligencia Artificial | 2025</small>
