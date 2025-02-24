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
                sh 'python3 -m pytest tests/test_data_preparation.py'
                sh 'python3 -m pytest tests/test_model_training.py'
                sh 'python3 -m pytest tests/test_model_evaluation.py'
                
            }
        }

        stage('Run Integration Tests') {
            steps {
                sh 'python3 -m pytest tests/test_integration.py'
            }
        }

        stage('Run Performance Tests') {
            steps {
                sh 'python3 -m pytest tests/test_performance.py'
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
    stage('Construire l\'Image Docker') {
            steps {
                sh 'docker build -t mon_projet_ml .'
            }
        }

        stage('Pousser l\'Image Docker') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'zeyneb', usernameVariable: 'zeyneb436', passwordVariable: 'zeyneb')]) {
                    sh 'docker login -u $DOCKER_USER -p $DOCKER_PASSWORD'
                    sh 'docker tag mon_projet_ml zeyneb436/mon_projet_ml:latest'
                    sh 'docker push zeyneb436/mon_projet_ml:latest'
                }
            }
        }
    }
    post {
    always {
        archiveArtifacts artifacts: 'models/*.joblib', allowEmptyArchive: true
        archiveArtifacts artifacts: 'logs/*.log', allowEmptyArchive: true
        archiveArtifacts artifacts: 'test-results/*.xml', allowEmptyArchive: true
    }

}