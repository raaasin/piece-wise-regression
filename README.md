# Piece Wise Regression

Piecewise linear regression-based algorithm is designed to predict continuous values using micro linear regressions. It segments the data into smaller linear sections and aggregates predictions from these sections to make an overall prediction.

## Overview

The algorithm utilizes mini linear regressions to divide the dataset into smaller segments based on the nearest 'X' values. It then predicts new values by considering the weighted average of these segments.

## Features

- Divides the dataset into smaller linear segments
- Utilizes mini linear regressions to predict values for new data points
- Allows users to adjust the number of nearest segments to consider for predictions
- Handles duplicate data by removing duplicate entries for accurate training

## Usage

### Installation

Clone the repository:

```bash
git clone https://github.com/raaasin/piece-wise-regression
```

### Usage Example

```python
from mini_linear_forest import MiniLinearForest

# Prepare your data
data = {'X': [1500, 2000, 1300, 1800, 2200, 1600, 1900, 2100, 1400, 2300, 1700, 2500, 2000, 1500, 1800],
        'Y': [300000, 400000, 250000, 350000, 420000, 280000, 370000, 410000, 260000, 450000, 320000, 480000, 390000, 290000, 340000]}

new_x = [1700, 1900, 2050, 2600]  # New X values to predict

# Create MiniLinearForest instance
forest = MiniLinearForest(nearest_segments=3, nearest_weight=0.5, rest_weight=0.7)

# Train the model
forest.train(data)

# Make predictions
predicted_prices = forest.predict(new_x)

# Plot predictions
forest.plot_predictions(data, new_x, predicted_prices)
```

## Parameters

- `nearest_segments`: Number of nearest segments to consider for predictions (default: 2)
- `nearest_weight`: Weight assigned to nearest segments for prediction aggregation (default: 0.4)
- `rest_weight`: Weight assigned to rest of the segments for prediction aggregation (default: 0.6)

## Contributions

Contributions and feature requests are welcome! Feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
"# reg-linear-forest" 
