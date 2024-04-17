import matplotlib.pyplot as plt
import seaborn as sns; sns.set() #for advanced plot styling - added by Kristi

# Original sklearn import
# from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# Kristi's updated sklearn import
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report

class MetricsAnalyzer ():
    def __init__(self, model, data, true_labels, pos_label=None) -> None:
        self.model = model
        self.data = data
        self.true_labels = true_labels
        self.pos_label = pos_label
        self.predictions = self.model.predict(self.data)

    def print_all (self, roc=True, plot_title=None):
        if roc:
            # print("ROC plot: ")
            self.plot_ROC(plot_title)
        #print("Confusion Matrix: ")
        self.print_confusion_matrix()
        print(classification_report(self.true_labels, self.predictions))
        '''
        print("Accuracy Score: ")
        self.print_accuracy()
        print("Recall Score: ")
        self.print_recall()
        print("Precision Score: ")
        self.print_precision()
        print("F1 Score: ")
        self.print_f1()
        '''
    '''
    # Original plot_ROC
    def plot_ROC (self, title = None):
        prob_scores = self.model.predict_proba(self.data)
        fpr, tpr, _ = roc_curve(self.true_labels, prob_scores[:, 1], pos_label=self.pos_label)
        if title:
            plt.title(title)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the plot
        plt.show()
    '''
    
    # Kristi's modified plot_ROC
    def plot_ROC (self, title = None):
        prob_scores = self.model.predict_proba(self.data)
        fpr, tpr, _ = roc_curve(self.true_labels, prob_scores[:, 1], pos_label=self.pos_label)
        roc_auc = auc(fpr, tpr) # Get the area under the curve

        if title:
            plt.title(title)

        # Plot the ROC curve
        plt.suptitle('ROC Curve')
        plt.plot(fpr, tpr, color='blue', label='Trained Classifier (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--', color='red', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
    
    def print_confusion_matrix (self):
        # Original print_confusion_matrix had only this line
        #print(confusion_matrix(self.true_labels, self.predictions))

        # Kristi's Replacement Code
        mat = confusion_matrix(self.true_labels, self.predictions)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=['climate','sports'],
                    yticklabels=['climate', 'sports'])
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
