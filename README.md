# SOANN: Seagull-Optimized Artificial Neural Network for Fault Prediction  

This project implements a **Seagull-Optimized Artificial Neural Network (SOANN)**, inspired by the paper *"A fault-tolerant model for cloud computing environment using seagull-optimized artificial neural network (SOANN)"*.  
Link: https://www.researchgate.net/publication/388833375_A_fault-tolerant_model_for_cloud_computing_environment_using_seagull-optimized_artificial_neural_network_SOANN

The model is tested on the **Backblaze Hard Drive Dataset**, which provides real-world SMART sensor attributes and failure logs for thousands of drives.  

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

## 🔹 What the Code Does
- Example Output (comparison):

================================================================================
COMPARISON: BASELINE ANN vs SOANN (paper replica)
================================================================================

ML METRICS:
accuracy   | ANN =  0.9971 | SOANN =  0.9982 | Δ = +0.0011
precision  | ANN =  0.3659 | SOANN =  0.6190 | Δ = +0.2532
recall     | ANN =  0.4688 | SOANN =  0.4062 | Δ = -0.0625
f1         | ANN =  0.4110 | SOANN =  0.4906 | Δ = +0.0796
roc_auc    | ANN =  0.9785 | SOANN =  0.9799 | Δ = +0.0014

SERVICE METRICS:
AST            | ANN =    2.95 | SOANN =    2.59 | Δ =   +0.36 (lower better)
throughput_rpm | ANN =  196.00 | SOANN =  196.00 | Δ =   +0.00
success_rate_pct | ANN =  100.00 | SOANN =  100.00 | Δ =   +0.00
availability_pct | ANN =    0.00 | SOANN =    0.00 | Δ =   +0.00

---

⚡ This repo shows how nature-inspired optimization (SOA) can enhance fault prediction models beyond traditional ANNs.







