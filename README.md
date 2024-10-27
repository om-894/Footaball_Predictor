# Football Match Predictor

This project is a deep learning model designed to predict **fouls committed** and **fouls won** for Sunderland AFC players in the next matchday. Using historical match data from previous weeks, the model averages relevant player statistics to make these predictions. Built with PyTorch, this project offers insights into player performance and match outcomes.

## Project Structure

- `data/`: Folder containing the Sunderland AFC match statistics data used for training and testing the model.
- `footy_data.py`: Script for data handling tasks, including downloading, cleaning, and preparing the football data.
- `main.py`: Main script to execute the workflow, from data loading and model training to generating predictions for the next matchday.
- `per_90_min_data_processing.py`: Processes statistics on a per-90-minute basis, normalizing player data to a standard 90-minute match format.
- `per_min_data_processing.py`: Similar to the per-90 script but handles per-minute processing, useful for players with fewer minutes.
- `README.md`: Project overview, setup instructions, and documentation.
- `requirements.txt`: List of all required Python packages. Install with `pip install -r requirements.txt`.

## Setup Instructions

To set up and run the project, follow these steps:

1. **Clone the Repository**  
   Clone this repository to your local machine.
   ```bash
   git clone https://github.com/yourusername/football-predictor.git
   cd football-predictor

2. Create a Virtual Environment
   It’s recommended to use a virtual environment to manage dependencies.
   ```bash
   python3 -m venv my_env

3. Activate the Virtual Environment
   On macOS/Linux:
   ```bash
   source my_env/bin/activate

4. Install Required Packages
   Use requirements.txt to install all dependencies.
   ```bash
   pip install -r requirements.txt

## Usage

1. **Prepare the Data**  
   Ensure the data files are in the `data/` folder. If the data is not included, refer to the data source instructions in this README.

2. **Run the Model**  
   To generate predictions for the next matchday, run the `main.py` script.
   ```bash
   python main.py

3. **Interpret Results**
The model will output predictions for fouls committed and fouls won for Sunderland AFC players based on their historical performance.

## Data Sources
The project uses historical performance data for Sunderland AFC players. Ensure that the data files are structured and formatted as expected by the scripts. If you're using external data, refer to the footy_data.py script for any preprocessing requirements.

## Example Output
Sample output for a player prediction might look like:

Player: Dan Neil
Predicted Fouls Committed: 1.2
Predicted Fouls Won: 0.8

## Requirements
The primary libraries used in this project include:

- `torch`
- `numpy`
- `pandas`
Refer to requirements.txt for the complete list.

## License
This project is licensed under the MIT License. See the LICENSE file for details.