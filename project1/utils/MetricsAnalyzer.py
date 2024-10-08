import matplotlib.pyplot as plt
import seaborn as sns; sns.set() #for advanced plot styling - added by Kristi
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report

class MetricsAnalyzer ():
    def __init__(self, model, data, true_labels, pos_label=None) -> None:
        self.model = model
        self.data = data
        self.true_labels = true_labels
        self.pos_label = pos_label
        self.predictions = self.model.predict(self.data)

    def print_all (self, roc=True, plot_title=None):
        print("Classification Measures: ", plot_title)
        print()
        print("Accuracy: ", plot_title)
        print(accuracy_score(self.true_labels, self.predictions))
        if roc:
            self.plot_ROC(plot_title)
            
        self.print_confusion_matrix(plot_title)
        print("Classification Report")
        if plot_title: print(plot_title)
        print()
        print(classification_report(self.true_labels, self.predictions, digits=4))
    
    def plot_ROC (self, title = None):
        prob_scores = self.model.predict_proba(self.data)
        fpr, tpr, _ = roc_curve(self.true_labels, prob_scores[:, 1], pos_label=self.pos_label)
        roc_auc = auc(fpr, tpr) # Get the area under the curve

        if title:
            plt.title(title)

        # Plot the ROC curve
        plt.suptitle('ROC Curve')
        plt.plot(fpr, tpr, color='blue', label='Trained Classifier (area = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--', color='red', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
    
    def print_confusion_matrix (self, title = None):
        mat = confusion_matrix(self.true_labels, self.predictions)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=['climate','sports'],
                    yticklabels=['climate', 'sports'])
        plt.suptitle("Confusion Matrix")
        if title: plt.title(title)
        plt.xlabel('True Label')
        plt.ylabel('Predicted Label')

    def get_accuracy (self):
        return accuracy_score(self.true_labels, self.predictions)

    def print_accuracy (self):
        print(accuracy_score(self.true_labels, self.predictions))

    def print_recall (self):
        print(recall_score(self.true_labels, self.predictions, pos_label=self.pos_label))

    def print_precision (self):
        print(precision_score(self.true_labels, self.predictions, pos_label=self.pos_label))

    def print_f1 (self) :
        print(f1_score(self.true_labels, self.predictions, pos_label=self.pos_label))
