pipeline {
    agent any
    environment {
        VENV_PATH = 'venv'
    }
    stages {
        stage('Checkout') {
            steps {
                git branch: 'master', url: 'https://github.com/zeynebnacef/mon_projet_ml.git'
            }
        }
        stage('Setup Python Environment') {
            steps {
                sh '''
                    python3 -m venv ${VENV_PATH}
                    . ${VENV_PATH}/bin/activate
                    python3 -m pip install --upgrade pip
                    python3 -m pip install --ignore-installed -r requirements.txt
                '''
            }
        }
        stage('Unit Tests') {
            steps {
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                    sh '''
                        . ${VENV_PATH}/bin/activate
                        export PYTHONPATH=${WORKSPACE}  # Ajouter le répertoire de travail au PYTHONPATH
                        pytest --cov=src --cov-report=xml --junitxml=pytest_report.xml test/
                    '''
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'pytest_report.xml, coverage.xml', allowEmptyArchive: true
                }
            }
        }
        stage('Prepare Data') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    if [ ! -f data/train.csv ] || [ ! -f data/test.csv ]; then
                        echo "Missing dataset files!"
                        exit 1
                    fi
                    python3 src/main.py --train-data data/train.csv --test data/test.csv --prepare
                '''
            }
        }
        stage('Train Model') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    python3 src/main.py --train-data data/train.csv --test data/test.csv --train
                '''
            }
        }
        stage('Evaluate Model') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    python3 src/main.py --train-data data/train.csv --test data/test.csv --evaluate
                '''
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