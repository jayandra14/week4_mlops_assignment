import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas
from matplotlib import pyplot as plt

model = joblib.load("./artifacts/models/v1/model.joblib")
data = pandas.read_csv("./data/iris.csv")

X = data[['sepal_length','sepal_width','petal_length','petal_width']]
y = data.species

y_pred = model.predict(X)

report = classification_report(y, y_pred, output_dict=True)

df = pandas.DataFrame(report).transpose()
df.to_csv("./metrics/report.csv", index_label='Metric')

cm = confusion_matrix(y, y_pred)

cmd = ConfusionMatrixDisplay(cm)

cmd.plot()
plt.savefig("./metrics/confusion_matrix")