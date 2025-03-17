# Customer Segmentation with K-Means Clustering (Google Colab)

This project focuses on segmenting retail store customers into distinct groups based on their purchasing behavior using the K-means clustering algorithm. It was developed to demonstrate the ability to analyze customer data, identify patterns, and derive actionable business insights. The project is designed to run on **Google Colab**, making it easy to execute without needing a local setup.

---

## Project Overview

The project involves the following steps:

1. **Data Loading and Exploration**: The dataset is loaded and analyzed to understand its structure and features.
2. **Data Preprocessing**:
   - Relevant features (`Annual Income`, `Spending Score`, and `Age`) are selected.
   - Features are standardized for optimal clustering performance.
3. **Optimal Cluster Analysis**: The Elbow Method and Silhouette Scores determine the ideal number of clusters.
4. **Model Training**: K-means clustering groups customers into segments.
5. **Visualization**: Results are visualized using 2D/3D plots to highlight customer groups.
6. **Cluster Interpretation**: Detailed analysis of cluster characteristics and actionable business recommendations.

---

## Dataset

The dataset is sourced from Kaggle:  
**[Customer Segmentation Tutorial Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)**  

It contains the following customer attributes:
- `CustomerID`: Unique identifier
- `Gender`: Male/Female
- `Age`: Customer’s age
- `Annual Income (k$)`: Yearly income
- `Spending Score (1-100)`: Spending behavior score  

Download the dataset from Kaggle and upload it to Google Colab for processing.

---

## How to Use (Google Colab)

1. **Open the Notebook in Google Colab**:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

2. **Upload the Dataset**:  
   - Download the dataset from Kaggle (`Mall_Customers.csv`).
   - Use the file browser in Google Colab to upload the dataset.

3. **Run the Notebook**:  
   Execute cells sequentially to:  
   - Preprocess the data  
   - Train the model  
   - Visualize clusters  
   - Generate business insights  

---

## Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `plotly` (for 3D visualization)

These libraries are pre-installed in Google Colab. No additional setup is required.

---

## Results

### 1. Optimal Cluster Analysis  
The Elbow Method and Silhouette Score analysis determine the optimal number of clusters (e.g., **5 clusters**):  
![Elbow Method](images/elbow_plot.png)  

---

### 2. Customer Segmentation Visualization  
Clusters visualized by `Annual Income` vs. `Spending Score`:  
![Clusters](images/clusters.png)  

---

### 3. Cluster Characteristics  
| Cluster | Avg. Age | Avg. Income | Avg. Spending Score | Target Strategy                     |  
|---------|----------|-------------|---------------------|-------------------------------------|  
| 0       | 32.7     | 86.5k       | 82.1                | Premium product campaigns           |  
| 1       | 25.5     | 26.5k       | 78.9                | Trend-focused promotions            |  
| 2       | 43.3     | 55.3k       | 49.2                | Loyalty programs                    |  
| 3       | 45.2     | 25.7k       | 20.3                | Budget-friendly deals               |  
| 4       | 41.1     | 88.2k       | 17.1                | Re-engagement campaigns             |  

---

## Business Recommendations

1. **Cluster 0 (High Income, High Spending)**: Target with luxury items and exclusive offers.  
2. **Cluster 1 (Young Moderate Spenders)**: Engage with social media campaigns.  
3. **Cluster 2 (Middle-Aged Average Spenders)**: Introduce loyalty rewards.  
4. **Cluster 3 (Budget-Conscious)**: Promote discounts and value bundles.  
5. **Cluster 4 (Low Engagement)**: Reactivate with personalized offers.  

---

## Contributing

Contributions are welcome! For improvements or bug fixes, please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset: [Kaggle Customer Segmentation Tutorial](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- Tools: `scikit-learn`, `matplotlib`, `pandas`

---

### Notes for Google Colab Users:
- Ensure `Mall_Customers.csv` is uploaded before running the notebook.
- Save a copy to Google Drive to retain changes.
