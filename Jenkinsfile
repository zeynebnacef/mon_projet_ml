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
                    // Generate artifacts in memory or a temporary directory
                    sh 'mkdir -p artifacts/models artifacts/test-results'
                    sh 'python3 -c "import joblib; joblib.dump({}, \'artifacts/models/gbm_model.joblib\')"'
                    sh 'echo "Test results" > artifacts/test-results/results.xml'
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
    }
}