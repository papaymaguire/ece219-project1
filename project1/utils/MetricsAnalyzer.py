import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
class MetricsAnalyzer ():
    def __init__(self, model, data, true_labels, pos_label=None) -> None:
        self.model = model
        self.data = data
        self.true_labels = true_labels
        self.pos_label = pos_label
        self.predictions = self.model.predict(self.data)

    def print_all (self):
        print("ROC plot: ")
        self.plot_ROC()
        print("Confusion Matrix: ")
        self.print_confusion_matrix()
        print("Accuracy Score: ")
        self.print_accuracy()
        print("Recall Score: ")
        self.print_recall()
        print("Precision Score: ")
        self.print_precision()
        print("F1 Score: ")
        self.print_f1()


    def plot_ROC (self):
        prob_scores = self.model.predict_proba(self.data)
        fpr, tpr, _ = roc_curve(self.true_labels, prob_scores[:, 1], pos_label=self.pos_label)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()

    def print_confusion_matrix (self):
        print(confusion_matrix(self.true_labels, self.predictions))

    def print_accuracy (self):
        print(accuracy_score(self.true_labels, self.predictions))

    def print_recall (self):
        print(recall_score(self.true_labels, self.predictions, pos_label=self.pos_label))

    def print_precision (self):
        print(precision_score(self.true_labels, self.predictions, pos_label=self.pos_label))

    def print_f1 (self) :
        print(f1_score(self.true_labels, self.predictions, pos_label=self.pos_label))