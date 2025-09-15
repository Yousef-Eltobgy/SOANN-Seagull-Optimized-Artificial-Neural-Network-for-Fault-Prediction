# SOANN: Seagull-Optimized Artificial Neural Network for Fault Prediction  

This project implements a **Seagull-Optimized Artificial Neural Network (SOANN)**, inspired by the paper *"A fault-tolerant model for cloud computing environment using seagull-optimized artificial neural network (SOANN)"*.  
Link: https://www.researchgate.net/publication/388833375_A_fault-tolerant_model_for_cloud_computing_environment_using_seagull-optimized_artificial_neural_network_SOANN

The model is tested on the **Backblaze Hard Drive Dataset**, which provides real-world SMART sensor attributes and failure logs for thousands of drives.  

---

⚡ This repo shows how nature-inspired optimization (SOA) can enhance fault prediction models beyond traditional ANNs.



---

## 🔹 What the Code Does
- **Baseline ANN:**  
  - A standard Artificial Neural Network trained on SMART attributes.  
  - Uses dropout and batch normalization to reduce overfitting.  
  - Evaluated on classification metrics (accuracy, precision, recall, F1, ROC AUC).  

- **SOANN (ANN + SOA):**  
  - Starts from the trained ANN.  
  - Applies the **Seagull Optimization Algorithm (SOA)** to optimize the **last layer weights**.  
  - Fine-tunes the last layer with gradient descent.  
  - Evaluated with the same ML metrics and cloud service simulation.  

- **Cloud Service Simulation (Paper-style):**  
  - Converts classification results into system-level metrics:  
    - **AST** (Average Service Time)  
    - **Throughput**  
    - **Success Rate**  
    - **Availability**  

- **Comparison Table:**  
  - Prints a side-by-side comparison between **ANN** and **SOANN** across both ML and simulation metrics.  
  - Saves results in `JSON` and model weights in `.pth` format.  


---

## 🔹 Why This Matters
- **Real-world impact:** Detects disk failures **before they happen**, improving reliability in cloud systems.  
- **SOA advantage:** The optimizer mimics seagull migration/spiral behavior, balancing **exploration vs exploitation**.  
- **Paper replication:** Matches the methodology from the referenced paper for reproducibility.  

---

## 🔹 How to Run
1. Download Backblaze dataset CSVs and place them in a `Data/` folder.  
   - [Backblaze HDD Dataset](https://www.backblaze.com/b2/hard-drive-test-data.html)  
2. Install dependencies:  
   ```bash
   pip install numpy pandas scikit-learn torch tqdm
3. Run the script

Results will be printed in the console and saved to:
- soann_paper_replica_results.json
- baseline_model.pth
- soann_model.pth


---

# 📊 COMPARISON: BASELINE ANN vs SOANN (Paper Replica)

- Example Output (comparison):
  - in this one, I used the data, from Backblaze drive_stats_2019_Q1.
  - I also set "USE_ALL_ATTRIBUTES = True" to use all the attributes not just the selected ones.


## ML Metrics
| Metric     | ANN     | SOANN   | Δ (Improvement) |
|------------|---------|---------|-----------------|
| Accuracy   | 0.9971  | 0.9982  | ✅ +0.0011       |
| Precision  | 0.3659  | 0.6190  | ✅ +0.2532       |
| Recall     | 0.4688  | 0.4062  | ⚠️ -0.0625       |
| F1 Score   | 0.4110  | 0.4906  | ✅ +0.0796       |
| ROC AUC    | 0.9785  | 0.9799  | ✅ +0.0014       |


## Service Metrics
| Metric            | ANN    | SOANN  | Δ (Improvement)        |
|-------------------|--------|--------|------------------------|
| AST               | 2.95   | 2.59   | ✅ +0.36 (lower better) |
| Throughput (rpm)  | 196.00 | 196.00 | ⚪ +0.00                |
| Success Rate (%)  | 100.00 | 100.00 | ⚪ +0.00                |
| Availability (%)  | 0.00   | 0.00   | ⚪ +0.00                |

✅ = improvement
⚠️ = drop
⚪ = no change



## 🔎 Summary
- **SOANN outperforms ANN** in most ML metrics (precision, F1 score, ROC AUC).  
- **Accuracy** shows a slight improvement.  
- **Recall** drops a bit, meaning SOANN catches fewer failures but with higher precision.  
- **Service metrics** show better **AST (Average Service Time)**, while throughput and success rate remain unchanged.  








