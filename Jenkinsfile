pipeline {
    agent any

    stages {
        stage('Hello') {
            steps {
                sh 'echo "Hello World"'
            }
        }
        stage('Checkout') {
            steps {
                checkout([$class: 'GitSCM', branches: [[name: '*/main']], userRemoteConfigs: [[url: 'https://github.com/yrmo/cudagrad.git']]])
            }
        }
        stage('Test CUDA') {
            steps {
                sh 'python project.py test CPU'
            }
        }
    }
}
