# import all the necessary packages for torch and torchvision

from sklearn.metrics import accuracy_score,f1_score,roc_auc_score, recall_score, precision_score
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_type", type=str, default='MME')
parser.add_argument("--model_type", type=str, default="ApexNet", help="(vit_L, vit_B, etc.)")
args = parser.parse_args()
results_path = 'results/predictions_{0}_{1}.csv'.format(args.model_type, args.data_type)
data = pd.read_csv(results_path)
predictions = data['predictions'].values
labels = data['labels'].values

val_f1_score = f1_score(labels,predictions,average='macro')
val_recall = recall_score(labels, predictions,zero_division = 0,average='macro')
val_precision = precision_score(labels,predictions, zero_division = 0,average='macro')
val_acc = accuracy_score(labels, predictions)

print('accuracy is: ', val_acc)
print('f1 score is: ', val_f1_score)
print('recall is: ', val_recall)
print('precision is: ', val_precision)

data = pd.DataFrame({'predictions':predictions, 'labels':labels})
data.to_excel('results_multi.xlsx')
    


    
