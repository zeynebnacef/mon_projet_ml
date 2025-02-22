pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/votre-utilisateur/votre-depot.git'
            }
        }

        stage('Prepare Data') {
            steps {
                sh 'python3 src/main.py --train data/train.csv --test data/test.csv --prepare'
            }
        }

        stage('Train Model') {
            steps {
                sh 'python3 src/main.py --train data/train.csv --test data/test.csv'
            }
        }

        stage('Evaluate Model') {
            steps {
                sh 'python3 src/main.py --train data/train.csv --test data/test.csv --evaluate'
            }
        }

        stage('Deploy Model') {
            steps {
                sh 'bash scripts/deploy.sh'
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
