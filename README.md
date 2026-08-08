# machine_learning

- Building the container: docker build -t rain-predictor-api .

- Running it: docker run -p 5000:5000 rain-predictor-api

# Current artifacts

Docker image: 
- ml_model_predictor_api_v_0.1

# Train model

python -m src.training.run_training

# For prediction after helm deployment:

kubectl port-forward service/machine-learning-machine-learning 5000:5000

# For new data

Run dvc add <filename> and dont have it ignored in the gitignore, the tracking works by itself