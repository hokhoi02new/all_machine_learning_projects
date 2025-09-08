from src.train import train_model
from src.evaluate import evaluate_model
import argparse
from src.predict import predict
def main():


    parser = argparse.ArgumentParser(description="California Housing Price Prediction")
    parser.add_argument("action", choices=["train", "evaluate", "predict"], help="Action to perform")
    parser.add_argument("--model", default="model.pkl", help="Path to save/load model")
    parser.add_argument("--model_type", default="linear", help="Model type: 'linear' or 'xgboost' or 'randomforest' or 'lightgbm' or 'stacking model'")
    parser.add_argument("--sample", nargs="+", type=float, help="Sample input for prediction")

    args = parser.parse_args()

    if args.action == "train":
        train_model(model_type=args.model_type, save_path=args.model)
    elif args.action == "evaluate":
        evaluate_model(model_file=args.model)
    elif args.action == "predict":
        if not args.sample:
            print("Please provide a sample input with --sample")
        else:
            predict(model_file=args.model, input_data=args.sample)


if __name__ == "__main__":
    main()