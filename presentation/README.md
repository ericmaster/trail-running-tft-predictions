# Trail Running TFT Predictor - Thesis Defense Presentation

## 📊 Presentation Overview

This presentation supports the thesis defense for "Application of Temporal Fusion Transformer to Trail Running Predictions" - Master's in Artificial Intelligence, Universidad San Francisco de Quito.

## 🚀 Quick Start

### Prerequisites
- Node.js (v16 or higher)
- npm

### Installation

```bash
cd presentation
npm install
```

### Running the Presentation

**Development mode (with live reload):**
```bash
npm run dev
```

**Production mode:**
```bash
npm start
```

The presentation will be available at `http://localhost:1948`

### Building Static Files

**Export to static HTML:**
```bash
npm run build
```

**Export to PDF:**
```bash
npm run build:pdf
```

**Export both HTML and PDF:**
```bash
npm run export
```

## 📁 Structure

```
presentation/
├── slides.md          # Main presentation content
├── package.json       # npm dependencies and scripts
├── README.md          # This file
└── dist/              # Generated static files (after build)
```

## 🎨 Presentation Features

- **16 slides** covering all key aspects of the thesis
- **Dark theme** (night) with gradient backgrounds
- **Responsive design** with proper scaling
- **Interactive elements** with reveal.js fragments
- **Mathematical equations** using MathJax
- **Tables and charts** for clear data presentation

## 📑 Slide Overview

1. **Title Slide** - Project introduction
2. **The Challenge** - Trail running complexity and why prediction matters
3. **Problem Statement** - Cold-start challenge definition
4. **Our Solution** - TFT architecture overview
5. **Key Contributions** - 6 main contributions
6. **Data Pipeline** - Distance-domain resampling
7. **Cold-Start Solution** - Synthetic encoder approach
8. **Asymmetric Loss** - Bias correction methodology
9. **Results V1 vs V2** - Quantitative comparison
10. **Visual Results** - Accumulated duration plots
11. **V3 Transfer Learning** - Garmin fine-tuning
12. **Limitations** - Honest assessment
13. **Applications** - Practical use cases
14. **Future Work** - Research directions
15. **Conclusions** - Key findings
16. **Q&A** - Thank you slide

## 🎯 Evaluation Criteria Addressed

### Clarity ✓
- Logical narrative flow from problem → solution → results → conclusions
- Clear visuals with gradient backgrounds and structured layouts
- Emoji icons for visual engagement
- Fragment animations for progressive disclosure

### Topic Domain ✓
- Demonstrates TFT architecture knowledge
- Explains cold-start methodology in depth
- Shows mathematical formulations
- Discusses error analysis and transfer learning

### Questions ✓
- Includes honest limitations section
- Provides quantitative evidence for claims
- Error cancellation insight shows deep understanding
- Future work demonstrates awareness of gaps

### Conclusions ✓
- Clear enumeration of 6 key findings
- Concrete solutions proposed (asymmetric loss, synthetic encoder)
- Actionable future directions
- Practical applications for athletes

## ⌨️ Keyboard Shortcuts (during presentation)

| Key | Action |
|-----|--------|
| `→` / `Space` | Next slide |
| `←` | Previous slide |
| `Esc` | Overview mode |
| `S` | Speaker notes |
| `F` | Fullscreen |
| `?` | Help |

## 🔧 Customization

To modify the presentation theme or options, edit the YAML frontmatter in `slides.md`:

```yaml
---
title: Your Title
theme: night  # Options: black, white, league, beige, sky, night, serif, simple, solarized
highlightTheme: monokai
revealOptions:
  transition: slide  # Options: none, fade, slide, convex, concave, zoom
  transitionSpeed: fast
  controls: true
  progress: true
---
```

## 📚 References

- [reveal-md documentation](https://github.com/webpro/reveal-md)
- [reveal.js documentation](https://revealjs.com/)
- [MathJax documentation](https://www.mathjax.org/)
