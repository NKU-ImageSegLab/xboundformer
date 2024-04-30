import csv
import os

import torchmetrics
import torch
from thop import profile

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MetricsResult:
    def __init__(self, result, mission="Binary"):
        self.F1 = result[f"metrics/{mission}F1Score"].item()
        self.Accuracy = result[f"metrics/{mission}Accuracy"].item()
        self.Dice = result[f"metrics/Dice"].item()
        self.Precision = result[f"metrics/{mission}Precision"].item()
        self.Specificity = result[f"metrics/{mission}Specificity"].item()
        self.Recall = result[f"metrics/{mission}Recall"].item()
        self.JaccardIndex = result[f"metrics/{mission}JaccardIndex"].item()
        try:
            self.AUROC = result[f"metrics/{mission}AUROC"].item()
            self.AveragePrecision = result[f"metrics/{mission}AveragePrecision"].item()
        except:
            self.AUROC = 0
            self.AveragePrecision = 0

    def to_result_csv(self, path, model_name, flops=0, params=0):
        first_create = os.path.exists(path)
        with open(os.path.join(path), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            if not first_create:
                wr.writerow([
                    'Model',
                    'Miou(Jaccard Similarity)', 'F1_score', 'Accuracy', 'Specificity',
                    'Sensitivity', 'DSC', 'Precision',
                    'AP', 'AUC', 'Parameters（M）', 'FLOPs(G)'
                ])

            wr.writerow([
                model_name,
                self.JaccardIndex * 100, self.F1 * 100, self.Accuracy * 100, self.Specificity * 100,
                self.Recall * 100, self.Dice * 100, self.Precision * 100,
                self.AveragePrecision * 100, self.AUROC * 100,
                params / 1e6, flops / 1e9
            ])

    def to_log(self, type, epoch, end_epoch, tr_loss):
        return f'Epoch [{epoch + 1}' \
               f'/{end_epoch}], Loss: {tr_loss:.4f}, \n' \
               f'[{type}] Acc: {self.Accuracy:.4f}, ' \
               f'SE: {self.Recall:.4f}, ' \
               f'SP: {self.Specificity:.4f}, ' \
               f'PC: {self.Precision:.4f}, ' \
               f'F1: {self.F1:.4f}, ' \
               f'DC: {self.Dice:.4f}, ' \
               f'MIOU: {self.JaccardIndex:.4f}'

    def cal_params_flops(self, model, size):
        input = torch.randn(1, 3, size, size).cuda()
        flops, params = profile(model, inputs=(input,))
        return params, flops


def get_metrics(number_classes):
    metrics = torchmetrics.MetricCollection(
        [
            torchmetrics.F1Score(
                task="multiclass", num_classes=2
            ),
            torchmetrics.Accuracy(
                task="multiclass", num_classes=2
            ),
            torchmetrics.Dice(),
            torchmetrics.Precision(
                task="multiclass", num_classes=2
            ),
            torchmetrics.Specificity(
                task="multiclass", num_classes=number_classes
            ),
            torchmetrics.Recall(
                task="multiclass", num_classes=number_classes
            ),
            # torchmetrics.AUROC(task="multiclass", number_classes=number_classes),
            # torchmetrics.AveragePrecision(task="multiclass", num_classes=number_classes),
            # IoU
            torchmetrics.JaccardIndex(
                task="multiclass", num_classes=number_classes
            ),
        ],
        prefix="metrics/",
    )
    # test_metrics
    test_metrics = metrics.clone(prefix="").to(device)
    return test_metrics


# {'metrics/BinaryF1Score': tensor(0.5072, device='cuda:0'), 'metrics/BinaryAccuracy': tensor(0.4850, device='cuda:0'),
#  'metrics/Dice': tensor(0.5072, device='cuda:0'), 'metrics/BinaryPrecision': tensor(0.4953, device='cuda:0'),
#  'metrics/BinarySpecificity': tensor(0.4490, device='cuda:0'), 'metrics/BinaryRecall': tensor(0.5196, device='cuda:0'),
#  'metrics/BinaryAUROC': tensor(0.4843, device='cuda:0'),
#  'metrics/BinaryAveragePrecision': tensor(0.5024, device='cuda:0'),
#  'metrics/BinaryJaccardIndex': tensor(0.3397, device='cuda:0')}


def get_binary_metrics(*args, **kwargs):
    metrics = torchmetrics.MetricCollection(
        [
            torchmetrics.F1Score(task="binary"),
            torchmetrics.Accuracy(task="binary"),
            torchmetrics.Dice(multiclass=False),
            torchmetrics.Precision(task="binary"),
            torchmetrics.Specificity(task="binary"),
            torchmetrics.Recall(task="binary"),
            torchmetrics.AUROC(task="binary"),
            torchmetrics.AveragePrecision(task="binary"),
            # IoU
            torchmetrics.JaccardIndex(task="binary", num_labels=2, num_classes=2),
        ],
        prefix="metrics/",
    )

    # test_metrics
    test_metrics = metrics.clone(prefix="").to(device)
    return test_metrics
