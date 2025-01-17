import pandas as pd
from sklearn.metrics import cohen_kappa_score

file_path = "C:\\Users\\jawad\\Downloads\\observerpair2-v2.xlsx"
data = pd.read_excel(file_path)

observer1 = data['Observer1'].fillna('None')
observer2 = data['Observer2'].fillna('None')

kappa = cohen_kappa_score(observer1, observer2)

print(f"Cohen's Kappa: {kappa}")
