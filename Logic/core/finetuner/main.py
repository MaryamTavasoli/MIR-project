# Instantiate the class
from Logic.core.finetuner.BertFinetuner_mask import BERTFinetuner
import wandb


def main():
    bert_finetuner = BERTFinetuner(
        '/Users/maryamtavasoli/Desktop/IMDb-IR-System-main-phase2/Logic/core/indexes/documents_index.json',
        top_n_genres=5)

    # Load the dataset
    bert_finetuner.load_dataset()

    # Preprocess genre distribution
    bert_finetuner.preprocess_genre_distribution()

    # Split the dataset
    bert_finetuner.split_dataset(test_size=0.1, val_size=0.1)

    # Fine-tune BERT model
    bert_finetuner.fine_tune_bert()

    # Compute metrics
    bert_finetuner.evaluate_model()

    # Save the model (optional)
    bert_finetuner.save_model('Movie_Genre_Classifier')


if __name__ == "__main__":
    main()
