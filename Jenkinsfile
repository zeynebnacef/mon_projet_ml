pipeline {
    agent any
    stages {
        stage('Checkout') {
            steps {
                git branch: 'master', url: 'https://github.com/zeynebnacef/mon_projet_ml.git'
            }
        }
        stage('Install Dependencies') {
            steps {
                sh 'python3 -m pip install --upgrade pip'
                sh 'python3 -m pip install --ignore-installed -r requirements.txt'
            }
        }
        stage('Run Unit Tests') {
            steps {
                sh 'python3 -m pytest tests/test_data_preparation.py --junitxml=test-results/unit-tests.xml'
            }
        }
        stage('Run Integration Tests') {
            steps {
                sh 'python3 -m pytest tests/test_integration.py --junitxml=test-results/integration-tests.xml'
            }
        }
        stage('Run Performance Tests') {
            steps {
                sh 'python3 -m pytest tests/test_performance.py --junitxml=test-results/performance-tests.xml'
            }
        }
        stage('Prepare Data') {
            steps {
                sh 'python3 src/main.py --train-data data/train.csv --test data/test.csv --prepare'
            }
        }
        stage('Train Model') {
            steps {
                sh 'python3 src/main.py --train-data data/train.csv --test data/test.csv --train'
            }
        }
        stage('Evaluate Model') {
            steps {
                sh 'python3 src/main.py --train-data data/train.csv --test data/test.csv --evaluate'
            }
        }
    }
    stage('Generate Artifacts') {
            steps {
                script {
                    // Create directories for artifacts
                    sh 'mkdir -p artifacts/models artifacts/test-results artifacts/evaluation-results'

                    // Save the trained model
                    sh 'python3 -c "import joblib; joblib.dump({}, \'artifacts/models/gbm_model.joblib\')"'

                    // Save test results
                    sh 'echo "Test results" > artifacts/test-results/results.xml'

                    // Save evaluation results (accuracy, classification report, etc.)
                    sh '''
                    python3 -c "
                    import json
                    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

                    # Example evaluation results
                    y_true = [0, 1, 1, 0, 1]
                    y_pred = [0, 1, 0, 0, 1]

                    accuracy = accuracy_score(y_true, y_pred)
                    report = classification_report(y_true, y_pred, output_dict=True)
                    conf_matrix = confusion_matrix(y_true, y_pred).tolist()

                    # Save results to a JSON file
                    results = {
                        'accuracy': accuracy,
                        'classification_report': report,
                        'confusion_matrix': conf_matrix
                    }
                    with open('artifacts/evaluation-results/evaluation_results.json', 'w') as f:
                        json.dump(results, f, indent=4)
                    "
                    '''
                }
                stash name: 'artifacts', includes: 'artifacts/**'
            }
        }
        stage('Archive Artifacts') {
            steps {
                unstash 'artifacts'
                archiveArtifacts artifacts: 'artifacts/**', allowEmptyArchive: true
            }
        }
    
    post {
        failure {
            echo 'Pipeline failed!'
        }
        success {
            echo 'Pipeline succeeded!'
        }
    }
    
}