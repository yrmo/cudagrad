pipeline {
    agent any

    environment {
        PATH = "/home/ryan/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin:/home/ryan/.local/bin:/home/ryan/.local/bin:${env.PATH}"
    }

    stages {
        stage('Hello') {
            steps {
                sh 'echo "Hello World"'
            }
        }
        stage('python') {
            steps {
                sh 'which python'
            }
        }
        stage('Checkout') {
            steps {
                checkout([$class: 'GitSCM', branches: [[name: '*/main']], userRemoteConfigs: [[url: 'https://github.com/yrmo/cudagrad.git']]])
            }
        }
        stage('pip') {
            steps {
                sh 'pip install -r /var/lib/jenkins/workspace/cudagrad/dev-requirements.txt'
            }
        }
        stage('Test CUDA') {
            steps {
                sh 'python project.py test CUDA'
            }
        }
    }
}
