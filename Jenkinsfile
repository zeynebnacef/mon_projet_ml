pipeline {
    agent any
    stages {
        // Stage 1: Checkout code
        stage('Checkout') {
            steps {
                git branch: 'master', url: 'https://github.com/zeynebnacef/mon_projet_ml.git'
            }
        }

        // Stage 2: Install dependencies
        stage('Install Dependencies') {
            steps {
                sh 'python3 -m pip install --upgrade pip'
                sh 'python3 -m pip install --ignore-installed -r requirements.txt'
            }
        }

        // Stage 3: Train the model and log to MLflow
        stage('Train Model and Log to MLflow') {
            steps {
                sh 'python3 src/main.py --train data/train.csv --test data/test.csv'
                echo 'Model trained and logged to MLflow'
            }
        }

        // Stage 4: Deploy Flask app
        stage('Deploy Flask App') {
            steps {
                // Stop any existing Flask app (if running)
                sh 'pkill -f "python3 app.py" || true'

                // Start the Flask app in the background
                sh 'nohup python3 app.py > flask.log 2>&1 &'

                // Print the Flask app logs
                sh 'cat flask.log'
                echo 'Flask app deployed and running on port 5005'
            }
        }

        // Stage 5: Wait for manual prediction
        stage('Wait for Manual Prediction') {
            steps {
                // Wait for user input
                input message: 'Please make a prediction using the Flask app. Click "Proceed" when done.', ok: 'Proceed'
                echo 'Prediction request sent manually.'
            }
        }

        // Stage 6: Verify prediction in MLflow
        stage('Verify Prediction in MLflow') {
            steps {
                script {
                    // Check MLflow for the latest prediction
                    def prediction = sh(script: 'mlflow runs search --experiment-name "Predictions" --order-by "attribute.start_time DESC" --max-results 1', returnStdout: true).trim()
                    if (prediction.contains("prediction")) {
                        echo 'Prediction successfully logged in MLflow!'
                    } else {
                        error 'Prediction not found in MLflow!'
                    }
                }
            }
        }
    }

    post {
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}