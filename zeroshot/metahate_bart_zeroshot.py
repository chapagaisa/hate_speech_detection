import pandas as pd
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection, metrics
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.multiprocessing as mp


import warnings
warnings.filterwarnings('ignore')

torch.multiprocessing.set_sharing_strategy('file_system')


class Config:
    VALID_BATCH_SIZE = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    TEST_DATA = "test.csv"
    model_id = "facebook/bart-large"
    pipeline = transformers.pipeline("zero-shot-classification", model=model_id, model_kwargs={"torch_dtype": torch.float16}, device=6)

class MetaHateDatSet(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path).fillna('none')
        self.data = self.data.reset_index(drop=True)
        self.text = self.data.text.values
        self.label = self.data.label.values

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        return {
            "text": self.text[item],
            "label": self.label[item]
        }

def classify_text(text):
    # Define candidate labels with explanations
    candidate_labels = ["Hate Speech", "Not Hate Speech"]
    response = Config.pipeline(text, candidate_labels = candidate_labels)
    #print(response)
    result = response['labels'][0] #1st label is higher score
    if result.startswith("Hate Speech"):
        return 1
    else:
        return 0

# Evaluation and ROC Curve
def save_roc_curve(y_true, test_probs, filename):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, test_probs)
    auc_score = metrics.roc_auc_score(y_true, test_probs)
    data = np.column_stack((fpr, tpr, thresholds))
    header = "False Positive Rate (FPR)\tTrue Positive Rate (TPR)\tThresholds"
    np.savetxt(filename + '.txt', data, header=header, delimiter='\t')
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.4f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename + '.png')
    plt.close()


def eval_fn(dataloader):
    fin_labels = []
    fin_outputs = []
    for d in tqdm(dataloader):
        text = d["text"]
        labels = d["label"]
        predictions = [classify_text(t) for t in text] 
        fin_labels.extend(labels)
        fin_outputs.extend(predictions)
    save_roc_curve(fin_labels, [prob for prob in fin_outputs], 'mh_roc_curve_bart_zeroshot')
    return fin_outputs, fin_labels

# Initialize datasets and dataloaders
test_data = MetaHateDatSet(Config.TEST_DATA)
test_dataloader = DataLoader(test_data, batch_size=Config.VALID_BATCH_SIZE, num_workers=4)


# Test evaluation
test_outputs, test_labels = eval_fn(test_dataloader)
test_accuracy = metrics.accuracy_score(test_labels, test_outputs)
print(f"Test Accuracy: {test_accuracy}")


print("==="*50)
print("\nClassification report \n\n", metrics.classification_report(test_labels, test_outputs, digits=4))

print("==="*50)
cm = metrics.confusion_matrix(test_labels, test_outputs)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
ax.set(xlabel="Predicted Label",
       ylabel="True Label",
       xticklabels=np.unique(test_labels),
       yticklabels=np.unique(test_labels),
       title="CONFUSION MATRIX")
plt.yticks(rotation=0)

# Save the confusion matrix plot
plt.savefig('mh_bart_zeroshot_confusion_matrix.png')
plt.close()
