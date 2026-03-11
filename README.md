# Customer Segmentation using K-Means

This project applies data preprocessing and machine learning techniques to segment customers based on their annual income and spending behavior.

The goal is to identify different customer groups that can help businesses better understand customer patterns and target their marketing strategies.

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib

---

## Project Steps

1. Data exploration and understanding the dataset  
2. Handling missing values using mode and median imputation  
3. Encoding categorical variables (Gender) using LabelEncoder  
4. Detecting and removing outliers using the IQR method  
5. Normalizing numerical features using MinMaxScaler and StandardScaler  
6. Performing correlation analysis using heatmap visualization  
7. Applying PCA for dimensionality reduction  
8. Using K-Means clustering to segment customers  

---

## Results

The customers were segmented into **five different clusters** based on their income and spending behavior.  
These clusters can help businesses identify:

- High value customers
- Budget customers
- Potential target customers

---

## How to Run the Project

1. Clone the repository

```bash
git clone https://github.com/khayal-ai/Customer-Segmentation.git
```

2. Navigate to the project folder

```bash
cd Customer-Segmentation
```

3. Install required libraries

```bash
pip install pandas scikit-learn seaborn matplotlib
```

4. Run the project

```bash
python main.py
```
