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

        // Stage 3: Run unit tests
        stage('Run Unit Tests') {
            steps {
                sh 'python3 -m pytest tests/test_data_preparation.py --junitxml=test-results/unit-tests.xml'
            }
        }

        // Stage 4: Run performance tests
        stage('Run Performance Tests') {
            steps {
                sh 'python3 -m pytest tests/test_performance.py --junitxml=test-results/performance-tests.xml'
            }
        }

        // Stage 5: Prepare data
        stage('Prepare Data') {
            steps {
                sh 'python3 src/main.py --train data/train.csv --test data/test.csv --prepare'
            }
        }

        // Stage 6: Train the model
        stage('Train Model') {
            steps {
                sh 'python3 src/main.py --train data/train.csv --test data/test.csv'
            }
        }

        // Stage 7: Evaluate the model
        stage('Evaluate Model') {
            steps {
                sh 'python3 src/main.py --train data/train.csv --test data/test.csv --evaluate'
            }
        }

        // Stage 8: Deploy Flask app
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

        // Stage 9: Test Flask app
        stage('Test Flask App') {
            steps {
                // Wait for the Flask app to start
                sh 'sleep 20'

                // Check if the Flask app is running
                sh 'ps aux | grep python3 app.py'

                // Send a test prediction request
                sh '''
                curl -X POST http://localhost:5005/predict -H "Content-Type: application/json" -d '{"features": [1, 2, 3, 4, 5, 6, 7, 8]}'
                '''

                // Check the Flask app logs
                sh 'cat flask.log'
                echo 'Prediction request sent and logged to MLflow'
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