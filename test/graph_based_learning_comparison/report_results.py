from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

def plot_acc(y_truth, y_pred, dataset_name, graph_type, classifier_type):
        # Get accuracy
        acc = accuracy_score(y_truth, y_pred)
        # Make confusion matrix
        conf = confusion_matrix(y_truth, y_pred)
        # Report Accuracy, type of graph
        plt.imshow(conf)
        plt.title(f"Confusion Matrix for {dataset_name} dataset based on {graph_type} graph with \n{classifier_type} classifier. Accuracy: {100*acc}%")