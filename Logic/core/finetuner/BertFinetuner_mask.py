import json
import torch
import wandb
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd


class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        # TODO: Implement initialization logic
        self.val_data = None
        self.test_data = None
        self.train_data = None
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.dataset = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=top_n_genres)
        self.label_encoder = LabelEncoder()

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        # TODO: Implement dataset loading logic
        with open(self.file_path, 'r') as file:
            data = json.load(file)['documents']
        summaries = []
        genres = []
        for doc in data.values():
            summaries.append(doc['first_page_summary'])
            if doc['genres']:
                genres.append(doc['genres'][0].lower())
            else:
                genres.append('unknown')
        self.dataset = pd.DataFrame({'text': summaries, 'labels': genres})
        print("Dataset loaded with {} entries.".format(len(self.dataset)))



    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        # TODO: Implement genre filtering and visualization logic
        self.dataset.dropna(subset=['text', 'labels'], inplace=True)
        label_counts = self.dataset['labels'].value_counts()
        top_labels = label_counts.nlargest(self.top_n_genres).index
        self.dataset = self.dataset[self.dataset['labels'].isin(top_labels)]
        self.dataset['labels'] = self.label_encoder.fit_transform(self.dataset['labels'])
        # Plot the genre distribution
        # genre_counts = self.dataset['genre'].value_counts()
        # genre_counts.plot(kind='bar')
        # plt.title('Top {} Genres Distribution'.format(self.top_n_genres))
        # plt.xlabel('Genre')
        # plt.ylabel('Count')
        # plt.show()
        # print("Top genres distribution plotted and dataset filtered.")

    def split_dataset(self, test_size=0.3, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        # TODO: Implement dataset splitting logic
        dataset_size = len(self.dataset)
        val_size = int(val_size * dataset_size)
        test_size = int(test_size * dataset_size)
        train_size = dataset_size - val_size - test_size
        train_data, test_data, val_data = random_split(
            self.dataset,
            [train_size, test_size, val_size]
        )
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        print("Dataset split into train, validation, and test sets.")

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        # TODO: Implement dataset creation logic
        return IMDbDataset(encodings, labels)

    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        # TODO: Implement BERT fine-tuning logic
        wandb.login(key='6bcfbb6b7bdfbcfe174074d1fa9b8a76576082ec')
        self._wandb = wandb
        self._wandb.init()
        train_summaries = [self.dataset.iloc[idx]['text'] for idx in self.train_data.indices if self.dataset.iloc[idx]['text'] is not None]
        train_labels = [self.dataset.iloc[idx]['labels'] for idx in self.train_data.indices if self.dataset.iloc[idx]['labels'] is not None]
        for t in train_summaries:
            if t is None:
                print("yes")
        train_encodings = self.tokenizer(train_summaries, truncation=True, padding=True)

        train_dataset = self.create_dataset(train_encodings, train_labels)
        val_summaries = [self.dataset.iloc[idx]['text'] for idx in self.val_data.indices if
                           self.dataset.iloc[idx]['text'] is not None]
        val_labels = [self.dataset.iloc[idx]['labels'] for idx in self.val_data.indices if
                        self.dataset.iloc[idx]['labels'] is not None]
        for t in val_summaries:
            if t is None:
                print("yes")
        val_encodings = self.tokenizer(val_summaries, truncation=True, padding=True)

        val_dataset = self.create_dataset(val_encodings, val_labels)
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir='./logs',
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        trainer.train()

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        # TODO: Implement metric computation logic
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        # TODO: Implement model evaluation logic
        test_summaries = [self.dataset.iloc[idx]['text'] for idx in self.test_data.indices if
                          self.dataset.iloc[idx]['text'] is not None]
        test_labels = [self.dataset.iloc[idx]['labels'] for idx in self.test_data.indices if
                       self.dataset.iloc[idx]['labels'] is not None]

        test_encodings = self.tokenizer(test_summaries, truncation=True, padding=True)
        test_dataset = IMDbDataset(test_encodings, test_labels)

        trainer = Trainer(
            model=self.model,
            compute_metrics=self.compute_metrics
        )

        eval_result = trainer.evaluate(test_dataset)
        print("Model evaluation results: ", eval_result)

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        # TODO: Implement model saving logic
        self.model.save_pretrained(model_name)
        self.tokenizer.save_pretrained(model_name)
        print(f"Model saved to {model_name} on Hugging Face Hub.")


class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        # TODO: Implement initialization logic

        # Print lengths for debugging
        print(f"Number of encodings: {len(list(encodings.values())[0])}")
        print(f"Number of labels: {len(labels)}")
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        # TODO: Implement item retrieval logic
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        # TODO: Implement length computation logic
        return len(self.labels)
