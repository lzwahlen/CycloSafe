# CycloSafe

## Setup:

use the following commands to create the conda environment: 

```bash
conda env create -f environment.yml
conda activate cyclosafe
```

## Temp notes on plots
 

### shap plot

Even though model prediction is not that good and F1 score is very low, I still used the SHAP plot. SHAP does not evaluate whether the model is accurate, but it explains what the model learned from the data it had. 

The model learned that cycleways are associated with higher predicted risk based on patterns in the training data.

These findings should be treated as preliminary until the model's predictive performance improves.

- each dot in plot represents one road semgent
- horizontal position = how strongly that feature pushed predicted risk score up (right) or down (left)
- dot color = segments actual feature value (e.g. for cycleways: red = 1 -> segment is cycleway, blue = 0 -> segment is not a cycleway)

Findings:
- highway_cycleway has many dots to the right, cycleways are associated with higher predicted risk, makes sense since there is more cycling in general
- junction_roundabout has one red dot at -0.45, being roundabout = very low predicted risk, makes sense since dutch roundabout separates cyclists from cars (safe)
- highway_tertiary has strongest positve val (0.15), signals strongest risk push, maybe because moderate speed with cars and bikes mixed -> dangerous
- highway_service has widest overall spread -> affects more segments than any other feature, but per-segment influence is weak
- violet dots in maxspeed: segments have medium speed limit, neither highest nor lowest in dataset -> present but maybe unreliable 