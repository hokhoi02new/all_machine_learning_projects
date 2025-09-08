from src.data_loader_process import load_data, process_data
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model, load_trained_model
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate fraud detection models")
    parser.add_argument("--model", type=str, default="random_forest",
        choices=["random_forest", "xgboost", "lightgbm", "neural_net", "svm", "naive_bayes"],
        help="Choose the model to train and evaluate"
        )
    args = parser.parse_args()

    model_name = args.model
    
    # Load raw data
    data = load_data()

    # Process data
    X_train, X_test, y_train, y_test = process_data(data)

    #build model
    if model_name == "neural_net":
        model = build_model(model_name, input_dim=X_train.shape[1])
    else:
        model = build_model(model_name)

    # Train & save
    train_model(model, X_train, y_train, model_name=model_name)

    # Load model back
    trained_model = load_trained_model(model_name)
    
    # Evaluate
    evaluate_model(trained_model, X_test, y_test)

if __name__ == "__main__":
    main()
    
