# Cold-Start Sequential Inference Summary

**Date:** 2025-12-09 01:43:11

## Configuration

- **Encoder Length:** 1 (minimal for cold-start)
- **Prediction Length per Chunk:** 200
- **Model Checkpoint:** ./checkpoints/best-checkpoint-epoch=17-val_loss=0.23.ckpt

## Session Information

- **Session ID:** training-session-2025-03-16-8073715633-14e5bad2-2106-4d13-a200-5643db7c17e1
- **Total Distance:** 24.51 km
- **Total Steps:** 4903
- **Number of Chunks:** 25

## Duration Prediction Results

| Metric | Value |
|--------|-------|
| Actual Total Duration | 324.93 min (5.42 h) |
| Predicted Total Duration | 229.95 min (3.83 h) |
| Absolute Error | 94.99 min |
| Percentage Error | 29.23% |

## Per-Chunk MAE (duration_diff)

| Chunk | Steps | MAE |
|-------|-------|-----|
| 1 | 0-200 | 0.5767 |
| 2 | 200-400 | 0.3648 |
| 3 | 400-600 | 0.8625 |
| 4 | 600-800 | 1.0232 |
| 5 | 800-1000 | 1.6126 |
| 6 | 1000-1200 | 2.0597 |
| 7 | 1200-1400 | 2.4550 |
| 8 | 1400-1600 | 0.7301 |
| 9 | 1600-1800 | 1.0084 |
| 10 | 1800-2000 | 0.1318 |
| 11 | 2000-2200 | 0.2630 |
| 12 | 2200-2400 | 3.8351 |
| 13 | 2400-2600 | 1.1062 |
| 14 | 2600-2800 | 0.8236 |
| 15 | 2800-3000 | 2.2715 |
| 16 | 3000-3200 | 1.8430 |
| 17 | 3200-3400 | 0.9313 |
| 18 | 3400-3600 | 0.9404 |
| 19 | 3600-3800 | 0.2200 |
| 20 | 3800-4000 | 3.3329 |
| 21 | 4000-4200 | 1.6538 |
| 22 | 4200-4400 | 2.5275 |
| 23 | 4400-4600 | 1.1884 |
| 24 | 4600-4800 | 1.1271 |
| 25 | 4800-4903 | 0.2434 |

## Error Evolution

| Progress | Distance (km) | Error (min) |
|----------|---------------|-------------|
| 25% | 6.1 | -19.83 |
| 50% | 12.2 | -42.74 |
| 75% | 18.4 | -65.16 |
| 100% | 24.5 | -94.99 |

## Methodology

1. **Cold-Start Initialization:** Synthetic encoder created using weighted average of first samples from all training sessions (chronological weighting)
2. **Known Terrain Data:** Actual altitude, elevation_diff, elevation_gain, elevation_loss used from target session (simulating GPS/route preview)
3. **Sequential Prediction:** Each chunk uses predicted values from previous chunks to inform the encoder for the next prediction
4. **Accumulated Duration:** Duration predictions (duration_diff) summed across all chunks

## Visualizations

- `cold_start_chunks.png`: Per-chunk duration_diff predictions vs actuals
- `cold_start_accumulated_error.png`: Accumulated duration and error over distance
