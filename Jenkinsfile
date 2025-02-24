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
                sh 'source venv/bin/activate && pip install -r requirements.txt'
            }
        }

        stage('Run Unit Tests') {
            steps {
                sh 'source venv/bin/activate && pytest tests/ --junitxml=test-results/unit-tests.xml'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'test-results/unit-tests.xml', allowEmptyArchive: true
                }
            }
        }

        stage('Prepare Data') {
            steps {
                sh 'source venv/bin/activate && python3 src/main.py --train-data data/train.csv --test data/test.csv --prepare'
            }
        }

        stage('Train Model') {
            steps {
                sh 'source venv/bin/activate && python3 src/main.py --train-data data/train.csv --test data/test.csv --train'
            }
        }

        stage('Evaluate Model') {
            steps {
                sh 'source venv/bin/activate && python3 src/main.py --train-data data/train.csv --test data/test.csv --evaluate'
            }
        }
    }

    post {
        failure {
            echo 'Pipeline failed! Check the logs for details.'
        }
        success {
            echo 'Pipeline succeeded!'
        }
    }
}