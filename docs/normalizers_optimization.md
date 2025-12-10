# Normalizers Optimization for Production Deployment

## Problem
The API initially required loading a 45MB `training_dataset_normalizers.pkl` file containing the full training dataset (352,205 samples) just to access the pre-fitted normalizers. This was necessary to avoid single-session normalization issues where GroupNormalizer would calculate statistics per session, resulting in constant predictions.

## Root Cause
PyTorch Forecasting's `GroupNormalizer` computes statistics (mean, std) per group:
- **Training**: Statistics calculated across 101 sessions → proper normalization
- **Single GPX session**: Only 1 session → variance = 0 → all values normalize to constant ~1.0

This caused predictions to be flat:
- Duration: 4.00s (constant)
- Heart Rate: 140.0 bpm (constant)  
- Cadence: 80.0 (constant)

## Solution Evolution

### v1: Load Full Training Dataset (45MB)
**File**: `checkpoints_v2/training_dataset_normalizers.pkl`
- Saved entire `TimeSeriesDataSet` with 352,205 samples
- Passed to `evaluate_full_session_sequential` via `training_dataset` parameter
- Used `TimeSeriesDataSet.from_dataset()` to copy normalizers to inference chunks
- **Size**: 45MB
- **Startup time**: Slow (loading large dataset)

### v2: Extract Only Normalizers (9.59 KB) ✅
**File**: `checkpoints_v2/normalizers.pkl`
- Extracted only essential normalization components:
  ```python
  {
      'target_normalizer': ...,           # Pre-fitted GroupNormalizer
      'categorical_encoders': ...,        # Encoders for categorical features
      'time_varying_known_reals': [...],  # Feature lists
      'time_varying_unknown_reals': [...],
      'target_names': [...],
      'group_ids': [...],
      'max_encoder_length': 400,
      'max_prediction_length': 200,
      # ... other config fields
  }
  ```
- **Size**: 9.59 KB (99.98% reduction!)
- **Startup time**: Fast (< 1 second)
- **No training data dependency**: Can deploy without training CSVs

## Implementation

### 1. Extract Normalizers (Run Once)
```bash
python utils/save_normalizers.py
```

This generates `checkpoints_v2/normalizers.pkl` containing:
- Pre-fitted `target_normalizer` (GroupNormalizer with statistics from 101 training sessions)
- Categorical encoders for session_id and other categorical features
- Dataset configuration (feature lists, encoder/prediction lengths, etc.)

### 2. Updated API (api/main.py)
```python
# Load lightweight normalizers at startup
NORMALIZERS_PATH = "checkpoints_v2/normalizers.pkl"
with open(NORMALIZERS_PATH, 'rb') as f:
    normalizers_data = pickle.load(f)

# Pass to inference function
result = evaluate_full_session_sequential(
    model=model,
    test_data=test_data,
    normalizers_data=normalizers_data,  # Instead of training_dataset
    ...
)
```

### 3. Updated Model Function (lib/model.py)
```python
def evaluate_full_session_sequential(
    model, 
    test_data,
    normalizers_data: dict = None,  # Changed from training_dataset
    ...
):
    if normalizers_data is not None:
        # Create TimeSeriesDataSet with pre-fitted normalizers
        chunk_dataset = TimeSeriesDataSet(
            chunk_data,
            time_idx="time_idx",
            target=normalizers_data['target_names'],
            target_normalizer=normalizers_data['target_normalizer'],  # Pre-fitted!
            categorical_encoders=normalizers_data['categorical_encoders'],
            time_varying_known_reals=normalizers_data['time_varying_known_reals'],
            time_varying_unknown_reals=normalizers_data['time_varying_unknown_reals'],
            ...
        )
```

## Results

### File Sizes
- **Before**: 45 MB (full training dataset)
- **After**: 9.59 KB (normalizers only)
- **Reduction**: 99.98%

### Predictions Quality
Both approaches produce identical predictions (verified on 17.8km GPX file):

| Metric | Range | Std Dev |
|--------|-------|---------|
| Duration | 2.09 - 9.96s | 1.34s |
| Heart Rate | 128.7 - 140.6 bpm | 2.31 bpm |
| Cadence | 55.4 - 85.0 | varies |

### Deployment Benefits
- ✅ **No training data dependency**: Only need model checkpoint + normalizers
- ✅ **Faster startup**: Loads in < 1 second
- ✅ **Smaller docker images**: 45MB → 9.59KB
- ✅ **Same prediction quality**: Identical normalization behavior
- ✅ **Production ready**: No need to ship training CSVs

## Files Modified
1. **Created**: `utils/save_normalizers.py` - Script to extract normalizers
2. **Modified**: `lib/model.py` - Changed parameter from `training_dataset` to `normalizers_data`
3. **Modified**: `api/main.py` - Load normalizers.pkl instead of training_dataset_normalizers.pkl
4. **Deleted**: Old `training_dataset_normalizers.pkl` (45MB) - no longer needed

## Migration Notes
- Old `training_dataset_normalizers.pkl` (45MB) can be deleted after generating `normalizers.pkl`
- Notebook workflows using `evaluate_full_session_sequential` continue to work (optional parameter)
- API now requires only: model checkpoint (476K params) + normalizers (9.59KB)
