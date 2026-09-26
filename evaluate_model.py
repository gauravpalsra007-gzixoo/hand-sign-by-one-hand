"""Report stratified dataset-holdout results for the default single-hand model."""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from feature_extraction import SINGLE_FEATURE_NAMES

def main():
    df=pd.read_csv("training_data.csv"); model=joblib.load("isl_model.pkl")
    X=df[SINGLE_FEATURE_NAMES]; y=df.label.astype(str)
    _,test=train_test_split(range(len(y)),test_size=.2,random_state=42,stratify=y)
    truth=y.iloc[test]; pred=model.predict(X.iloc[test]); labels=sorted(y.unique())
    print(f"Dataset holdout accuracy: {accuracy_score(truth,pred):.3%} (not live webcam accuracy)")
    print(classification_report(truth,pred,labels=labels,zero_division=0,digits=4))
    report=classification_report(truth,pred,labels=labels,output_dict=True,zero_division=0)
    pred_series=pd.Series(pred,index=truth.index)
    rows=[]
    for letter in labels:
        mask=truth==letter
        rows.append({"letter":letter,"accuracy":float((pred_series[mask]==truth[mask]).mean()) if mask.any() else 0.,
                     "precision":report[letter]["precision"],"recall":report[letter]["recall"],
                     "f1":report[letter]["f1-score"],"support":int(report[letter]["support"])})
    pd.DataFrame(rows).to_csv("per_letter_accuracy.csv",index=False)
    ConfusionMatrixDisplay(confusion_matrix(truth,pred,labels=labels),display_labels=labels).plot(cmap="Blues",xticks_rotation=45,values_format="d")
    plt.title("ISL A-Z Confusion Matrix — Dataset Holdout"); plt.tight_layout(); plt.savefig("confusion_matrix.png",dpi=180); plt.close()
    print("Saved confusion_matrix.png and per_letter_accuracy.csv")

if __name__=="__main__": main()
