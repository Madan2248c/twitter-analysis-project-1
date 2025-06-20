# Twitter Analysis Project

This project is designed to analyze Twitter data using various data science techniques. It includes data preprocessing, sentiment analysis, graph construction, and visualization of results. The analysis is presented through an interactive dashboard built with Streamlit.

## Project Structure

```
twitter-analysis-project/
├── data/
│   └── twitter_data.json          # Contains the Twitter data in JSON format.
├── notebooks/
│   └── temporal_graph_analysis.ipynb # Jupyter notebook for temporal graph analysis.
├── src/
│   ├── preprocess.py               # Functions for data preprocessing.
│   ├── build_graph.py              # Functions to construct a graph from the data.
│   ├── sentiment_analysis.py        # Functions for sentiment analysis.
│   └── visualization.py             # Functions for visualizing analysis results.
├── dashboard/
│   └── streamlit_app.py            # Main entry point for the Streamlit dashboard.
├── requirements.txt                 # Lists the Python dependencies required for the project.
└── README.md                        # Documentation for the project.
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/twitter-analysis-project.git
   cd twitter-analysis-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your Twitter data in the `data/twitter_data.json` file.
2. Run the preprocessing script to clean and transform the data:
   ```
   python src/preprocess.py
   ```

3. Build the graph from the preprocessed data:
   ```
   python src/build_graph.py
   ```

4. Perform sentiment analysis:
   ```
   python src/sentiment_analysis.py
   ```

5. Visualize the results:
   ```
   python src/visualization.py
   ```

6. Launch the Streamlit dashboard:
   ```
   streamlit run dashboard/streamlit_app.py
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.