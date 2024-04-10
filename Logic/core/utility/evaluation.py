import math
from typing import List
import wandb


class Evaluation:

    def __init__(self, name: str):
        self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        precision = 0.0

        # TODO: Calculate precision here
        num_samples = len(actual)
        sum_precision = 0.0

        for i in range(num_samples):
            intersection = len(set(actual[i]).intersection(set(predicted[i])))
            sum_precision += intersection / len(predicted[i])

        precision = sum_precision / num_samples

        return precision

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        recall = 0.0

        # TODO: Calculate recall here
        if not actual or not predicted:
            return recall

        num_samples = len(actual)
        sum_recall = 0.0

        for i in range(num_samples):
            intersection = len(set(actual[i]).intersection(set(predicted[i])))
            sum_recall += intersection / len(actual[i])

        recall = sum_recall / num_samples

        return recall

    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        f1 = 0.0

        # TODO: Calculate F1 here
        if not actual or not predicted:
            return f1

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)

        if precision + recall != 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        return f1

    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float | list[float]:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        AP = []
        if not actual or not predicted:
            return AP
        num_samples = len(actual)

        # TODO: Calculate AP here
        for i in range(num_samples):
            num_correct = 0
            sum_precision = 0.0

            for j in range(len(predicted[i])):
                if predicted[i][j] in actual[i]:
                    num_correct += 1
                    sum_precision += num_correct / (j + 1)

            if num_correct != 0:
                AP.append(sum_precision / len(actual[i]))

        return AP

    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        MAP = 0.0

        # TODO: Calculate MAP here
        AP = self.calculate_AP(actual, predicted)

        if not AP:
            return MAP

        num_queries = len(AP)
        sum_AP = sum(AP)
        MAP = sum_AP / num_queries
        return MAP

    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        DCG = 0.0

        # TODO: Calculate DCG here
        DCG = 0.0

        if not actual or not predicted:
            return DCG

        def DCG_at_k(scores, k):
            return

        num_samples = len(actual)
        sum_DCG = 0.0

        for i in range(num_samples):
            dcg_scores = [1 if item in actual[i] else 0 for item in predicted[i]]
            sum_DCG += sum((dcg_scores[j]) / (math.log2(j + 1)) for j in range(len(predicted[i])))

        DCG = sum_DCG / num_samples
        return DCG

    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCG = 0.0

        # TODO: Calculate NDCG here
        if not actual or not predicted:
            return NDCG

        def DCG_at_k(scores1, scores2):
            return sum((scores1[i] / scores2[i]) / (math.log2(i + 1)) for i in range(len(scores1)))

        num_samples = len(actual)
        sum_NDCG = 0.0

        for i in range(num_samples):
            actual_scores = [1 if item in actual[i] else 0 for item in predicted[i]]
            ideal_scores = sorted(actual_scores, reverse=True)
            dcg_scores = [1 if item in actual[i] else 0 for item in predicted[i]]
            sum_NDCG += DCG_at_k(dcg_scores, ideal_scores)

        NDCG = sum_NDCG / num_samples

        return NDCG

    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        RR = []
        if not actual or not predicted:
            return RR

        # TODO: Calculate MRR here
        num_samples = len(actual)

        for i in range(num_samples):
            for j in range(len(predicted[i])):
                if predicted[i][j] in actual[i]:
                    RR.append(1 / (j + 1))
                    break
        return RR

    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        MRR = 0.0
        RR = self.cacluate_RR(actual, predicted)

        if not RR:
            return MRR

        num_queries = len(RR)
        sum_RR = sum(RR)
        MRR = sum_RR / num_queries

        return MRR

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")

        # TODO: Print the evaluation metrics
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Average Precision: {ap}")
        print(f"Mean Average Precision: {map}")
        print(f"Discounted Cumulative Gain: {dcg}")
        print(f"Normalized Discounted Cumulative Gain: {ndcg}")
        print(f"Reciprocal Rank: {rr}")
        print(f"Mean Reciprocal Rank: {mrr}")

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """

        # TODO: Log the evaluation metrics using Wandb
        wandb.init(project='your_project_name', name=self.name)

        # Log the evaluation metrics using Wandb
        wandb.log({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap": ap,
            "map": map,
            "dcg": dcg,
            "ndcg": ndcg,
            "rr": rr,
            "mrr": mrr
        })

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        # call print and viualize functions
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
