# K-Means Clustering

A simple, end-to-end machine learning workflow demonstrating unsupervised clustering using **K-Means** in Python. 

This project walks through generating synthetic data, applying feature scaling, using the **Elbow Method** to find the optimal number of clusters ($k$), and visualizing the final customer/data segments using `seaborn` and `matplotlib`.

---

## What's in this Repo?

* **`k-mean_clustering.ipynb`**: The main Jupyter Notebook containing step-by-step code, model evaluation, and visualizations.

---

## Workflow & Methodology

### 1. Data Generation & Preprocessing
Instead of relying on an external CSV, we generate a controlled dataset using `scikit-learn`'s `make_blobs`. This creates **500 samples** distributed across **3 underlying centers** with a standard deviation of `0.60`.
* **Why scale?** Distance-based algorithms like K-Means are sensitive to the scale of input features. Before clustering, the dataset is normalized using `StandardScaler` so each feature contributes equally to the distance metrics.

### 2. Finding the Optimal $k$ (The Elbow Method)
To avoid guessing the number of clusters, we iterate over $k \in [1, 10]$ and calculate the **inertia** (Sum of Squared Errors within clusters). 

Plotting inertia against $k$ reveals a distinct drop that levels off—forming an "elbow"—at **$k = 3$**.

<!-- Tip: Save your elbow plot from Out[9] as images/elbow_plot.png and link it here -->
![Elbow Method Plot](path/to/your/elbow_plot.png) 
*Notice how the inertia drops significantly up to k=3, after which adding more clusters yields minimal returns.*

### 3. Training & Visualization
Using our optimal $k=3$, we fit the final `KMeans` model, assign cluster labels back to the original dataframe, and map the results. 

Using the `viridis` color palette, the scatter plot clearly separates our 500 data points into three distinct, non-overlapping groupings:

<!-- Tip: Save your scatter plot from Out[13] as images/cluster_results.png and link it here -->
![K-Means Clustering Results](path/to/your/cluster_results.png)

---

## Getting Started

### Prerequisites
Make sure you have Python installed along with the essential data science libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn jupyter