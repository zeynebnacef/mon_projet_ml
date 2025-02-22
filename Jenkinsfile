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
        stage('Prepare Data') {
            steps {
                sh 'python3 src/main.py --train data/train.csv --test data/test.csv --prepare'
            }
        }
        stage('Train Model') {
       	    steps {
                sh 'python3 src/main.py --train-data data/train.csv --test data/test.csv --train'
           }
        }
        stage('Evaluate Model') {
            steps {
                sh 'python3 src/main.py --test data/test.csv --evaluate'
            }
        }
        stage('Deploy Model') {
            steps {
                sh 'python3 src/main.py --deploy'
            }
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
