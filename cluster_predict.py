import os
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
from sklearn.decomposition import PCA 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

train_dataset_file = "dataset.csv"
if os.path.exists(train_dataset_file):
    df = pd.read_csv(train_dataset_file, usecols= ["Category", "News", "Cluster"])
else:
    df = pd.DataFrame(columns = ["Category", "News", "Cluster"])

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["News"])

# reduce the dimensionality of the data using PCA 
pca = PCA(n_components=2) 
reduced_data = pca.fit_transform(X.toarray()) 

num_clusters = 3

kmeans = KMeans(n_clusters=num_clusters, n_init=5, 
                max_iter=500, random_state=42)
kmeans.fit(X)

def predict_cluster(user_sentence):
    user_sentence_vector = vectorizer.transform([user_sentence])
    predicted_cluster = kmeans.predict(user_sentence_vector)[0]
    predicted_label = cluster_labels[predicted_cluster]

    return predicted_label, predicted_cluster

def on_predict_click(root):
    global df
    user_sentence = document_entry.get()
    
    predicted_label, predicted_cluster = predict_cluster(user_sentence)

    if predicted_label is not None:
        result_label.configure(text= "Predicted Cluster: " + predicted_label)
        messagebox.showinfo("Prediction Result", f"Predicted Cluster:{predicted_label}")
        
        # if df[df["News"] == user_sentence].empty:
        #     new_row = {"Category": predicted_label, "News": user_sentence, "Cluster" : predicted_cluster}
        #     df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=False)

        #     df.to_csv(train_dataset_file, index = False)

        # else:
        #     messagebox.showinfo("Duplicate Sentence", "The sentence already exists")
    else:
        messagebox.showerror("Prediction Error", "Failed to predict the cluster")

cluster_labels = {0: "Sport", 1: "Health", 2: "Business"}

root = tk.Tk()
root.title("Document Clustering")
root.geometry('500x400')

document_label = ttk.Label(root, text = "Enter a Sentence:")
document_entry = ttk.Entry(root, width = 50)
document_button = ttk.Button(root, text = "Predict Cluster", command = lambda: on_predict_click(root))
result_label = ttk.Label(root, text = "Predicted Cluster:", font=("Arial", 15))

document_label.pack(pady = 5)
document_entry.pack(pady = 5)
document_button.pack(pady = 10) 
result_label.pack(pady = 5)
root.mainloop()
