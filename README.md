# F1Predictor

F1Predictor is a Python-based machine learning application designed to predict the winner of Formula 1 races based on various race details and driver statistics.

## Features

- Predict race winners using historical data and advanced machine learning models.
- Provides insights into feature importance for predictions.
- Easy-to-use interface for entering race data like grid positions, track, fastest laps, etc.

## Getting Started

Follow the steps below to set up and run the repository.

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package installer)

Install the required Python libraries by running:

```bash
pip install -r requirements.txt
```

### Installation

Clone the repository:

```bash
git clone https://github.com/madtitan1712/F1Predictor.git
cd F1Predictor
```

### How to Run

#### 1. Train the Model
Run the `main.py` file to train the machine learning model using the cleaned dataset:

```bash
python main.py
```

This will generate a trained model file named `f1_prediction_model.pkl` in the `Model` directory.

#### 2. Predict Race Winner
Use `predictor.py` to predict race winners. Enter grid positions and track details to make predictions:

```bash
python predictor.py
```

Follow the on-screen instructions to input data.

### Directory Structure

```
F1Predictor/
├── DataClean.py          # Script to clean raw data
├── DataFarmer.py         # Script to collect raw data
├── F1Dataset.csv         # Raw data file
├── F1DatasetCleaned.csv  # Cleaned data file
├── Model/                # Directory for trained model files
├── predictor.py          # Script for making predictions
├── requirements.txt      # Python dependencies
├── encoding_mappings.json # Encoded mappings for drivers, teams, and tracks
├── main.py               # Training script
```

### Example Prediction
Here is an example of how the tool predicts race results:

1. Run `predictor.py`.
2. Enter the track name (e.g., "Monaco").
3. Enter grid positions (e.g., `VER 1`, `HAM 2`, `LEC 3`).
4. View the predicted probabilities for each driver to win the race.

## License

This repository is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the functionality.

## Acknowledgements
  -FastF1 for providing the F1 data access
  -Formula 1 for the incredible sport that inspires this project

## Contact

Got questions or suggestions? Feel free to reach out:
GitHub: @madtitan1712

Note: This project is for educational purposes and is not affiliated with Formula 1.
