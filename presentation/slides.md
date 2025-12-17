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
  math:
    mathjax: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
    config: 'TeX-AMS_HTML-full'
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

## 💡 The Solution

### Temporal Fusion Transformer (TFT)

<div style="display: flex; justify-content: space-around; align-items: flex-start;">
<div style="flex: 1;">

**Architecture Components:**
<div style="text-align: start; justify-content: flex-start;">
<div class="fragment">
- 🚪 Gated Residual Networks (GRN)
</div>
<div class="fragment">
- 📊 Variable Selection Networks
</div>
<div class="fragment">
- 🔄 LSTM Encoder-Decoder
</div>
<div class="fragment">
- 🎯 Multi-head Self-Attention
</div>
</div>

----

<!-- .slide: data-background-gradient="linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%)" -->

### Key Advantages

- Handles **known** vs **unknown** futures
- Learns **long-range** dependencies
- Provides **interpretable** attention weights
- Supports **multi-target** forecasting

</div>
</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #16213e 0%, #1a1a2e 100%)" -->

## 🔬 Contributions

<div style="font-size: 0.9em;">

1. **Novel Application**: First documented TFT for trail running
2. **Cold-Start Methodology**: Synthetic encoder approach
3. **Asymmetric Loss Function**: Corrects under-prediction bias
4. **Distance-Domain Resampling**: Pace-independent predictions
5. **Multi-Target Forecasting**: Duration, HR, temp, cadence
6. **Error Cancellation Analysis**: Robust evaluation guidance

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #0f3460 0%, #16213e 100%)" -->

## 📊 Data Pipeline

### Distance-Domain Resampling

```
106 Polar sessions (79 train / 16 val / 11 test)
Time Domain (1 sec) → Distance Domain (5 meters)
7 Garmin sessions (5 train / 1 val / 1 test)
```

<div style="display: flex; justify-content: space-around;">
<div class="fragment" style="flex: 1;">

**Input Features:**
- Heart Rate
- Altitude
- Cadence
- Speed
- Temperature

</div>
<div class="fragment" style="flex: 1;">

**Derived Features:**
- Elevation diff/gain/loss
- Duration per interval
- Fatigue proxies
- Rolling averages

</div>
</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)" -->

## 🧊 Cold-Start Solution

### Synthetic Encoder Approach

$$x\_{synthetic} = \\frac{\\sum\_{s=1}^{S} w\_s \\cdot x\_{s,0}}{\\sum\_{s=1}^{S} w\_s}$$

<div style="display: flex; justify-content: space-around; align-items: center;">

- Weight historical first samples
- Recent sessions weighted higher
- Use actual **terrain data** (known)
- Estimate physiological baseline

</div>

----

<!-- .slide: data-background-gradient="linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)" -->

### Synthetic Encoder Purpose

<div style="flex: 1;">

- Capture current **fitness level**
- Leverage **population patterns**
- GPS route **preview** available
- Add an initial input without any previous race data! 🎯

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)" -->

### 📊 Results

#### Accumulated Duration Prediction

![V1 vs V2 Comparison](assets/v1_v2_comparison.png)

<small>V2 (asymmetric SMAPE) shows substantially reduced under-prediction bias throughout the entire session</small>

----

<!-- .slide: data-background-gradient="linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%)" -->

### 📈 Results: V1 vs V2

#### Cold-Start on 24.5 km Session / 324.9 min

| Metric | V1 (sMAPE) | V2 (Asym) | Change |
|--------|------------|-----------------|--------|
| MAE (s/5m) | 1.347 | 1.066 | **-20.9%** |
| Bias (s/5m) | -1.162 | +0.157 | **+113.5%** |
| Predicted | 229.9 min | 337.7 min | +46.9% |
| **Accumulated Error** | **-29.2%** | **+3.9%** | ✅ |

<div class="fragment">

**Across all 11 test sessions:** -30.4% → **+3.7%** 🎯

</div>

----

<!-- .slide: data-background-gradient="linear-gradient(135deg, #16213e 0%, #0f3460 100%)" -->

### ⚖️ Asymmetric Loss Function

#### Correcting Under-Prediction Bias

$$\mathcal{L}_{asym} = w \cdot \frac{|y - \hat{y}|}{|y| + |\hat{y}| + \epsilon} \cdot 2$$

<div class="fragment">
$$
w =
\begin{cases}
\alpha & \text{if } y > \hat{y} \quad \text{(under-prediction)} \\\\
1 - \alpha & \text{if } y \leq \hat{y} \quad \text{(over-prediction)}
\end{cases}
$$

</div>

<div class="fragment">

**With α = 0.51:** Slight penalty for under-prediction

> ⚠️ *Highly sensitive parameter:* α>=0.55 caused high over-prediction

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #16213e 0%, #0f3460 100%)" -->

### 🔄 V3: Transfer Learning

#### Fine-tuning with Garmin + Nutrition Data

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

### ⚠️ Limitations

<div style="font-size: 0.85em;">

| Limitation | Impact | Mitigation and Needs |
|------------|--------|------------|
| **Single-Athlete Dataset** | Multi-athlete generalization | Multi-athlete data needed |
| **106 Sessions** | Overfitting risk | Dropout=0.25 / More data needed |
| **Geographic Specificity** | Andes-trained mostly | More data needed |
| **Missing Features** | Weather, sleep, HRV absent | Future sensor integration |
| **Fine Tuning Limitation** | Restricted to base model size | Increase model complexity |

</div>

<div class="fragment">

**V3 Lesson:** 5 sessions insufficient for sparse feature learning → **20+ sessions** at least

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%)" -->

### Practical Applications

<div style="display: flex; justify-content: space-around; flex-wrap: wrap;">

<div class="fragment" style="width: 75%; margin: 10px; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;">

#### 🏃 Race Planning
Estimate finish time **before race start** based on route profile

</div>

<div class="fragment" style="width: 75%; margin: 10px; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;">

#### 🍎 Nutrition Planning
Predicted duration informs **caloric & fluid** needs

</div>
</div>

----

<!-- .slide: data-background-gradient="linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%)" -->

### Practical Applications

<div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
<div class="fragment" style="width: 75%; margin: 10px; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;">

#### 📈 Training Optimization
Analyze **predicted vs actual** performance

</div>

<div class="fragment" style="width: 75%; margin: 10px; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;">

#### ⏱️ Pacing Strategy
Real-time predictions for **pacing adjustments**

</div>

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #16213e 0%, #1a1a2e 100%)" -->

## 🔮 Future Work

<div style="font-size: 0.9em;">

1. **Multi-Athlete Data** – Incorporate data from Strava and Garmin Connect to improve generalization.
2. **Few-Shot Analysis** – Evaluate predictions when a small amount of prior session data is available.
3. **Weather Integration** – Include external factors such as temperature, humidity, and wind.
4. **Uncertainty Quantification** – Provide confidence intervals for model predictions.
5. **Interpretability** – Analyze attention weights to better understand feature importance.

----

<!-- .slide: data-background-gradient="linear-gradient(135deg, #16213e 0%, #1a1a2e 100%)" -->

6. **Model Compression** – Enable on-device inference, e.g., from phone to watch.
7. **Multi-Metric Evaluation** – Assess both per-step and cumulative prediction accuracy.
8. **Session Type Classification** – Distinguish between training and race sessions.
9. **Race Planning Optimization** – Develop strategies to minimize race completion time.
10. **IMU Sensor Integration** – Incorporate preprocessed accelerometer/gyroscope data for terrain technicality analysis.

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #0f3460 0%, #16213e 100%)" -->

## 🎓 Conclusions

<div style="font-size: 0.8em;">

1. **TFT Successfully Applied** to trail running prediction
2. **Cold-Start Prediction Achieved** via synthetic encoder approach
3. **Asymmetric Loss** reduced error from **-30.4% to +3.7%**
4. **Distance-Domain Processing** enables pace-independent predictions
5. **Transfer Learning Works** - V2 model transferred to Garmin but more data is needed for true validation
6. **Critical Insight:** Evaluate both **per-step** and **cumulative** metrics

</div>

<div class="fragment" style="margin-top: 20px; font-size: 1.2em;">

> 🎯 *"Conservative over-prediction is preferable for race planning"*

</div>

---

<!-- .slide: data-background-gradient="linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)" -->

# Thank You!

### Questions?

<div style="margin-top: 50px;">

<div style="margin: 0 auto; width: 50%">

![Demo](assets/qrcode.png)
</div>

</div>

<small style="margin-top: 50px; display: block;">Universidad San Francisco de Quito | Maestría en Inteligencia Artificial | 2025</small>
