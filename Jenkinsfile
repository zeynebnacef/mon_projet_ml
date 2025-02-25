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
        post {
        failure {
            echo 'Pipeline failed!'
        }
        success {
            echo 'Pipeline succeeded!'
        }
    }
}