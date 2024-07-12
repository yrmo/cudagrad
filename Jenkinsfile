pipeline {
    agent any

    options { 
        disableConcurrentBuilds() 
    }

    environment {
        PATH = "/home/ryan/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin:/home/ryan/.local/bin:/home/ryan/.local/bin:${env.PATH}"
    }

    stages {
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
        stage('Post checkout') {
            steps {
                dir('/var/lib/jenkins/workspace/cudagrad') {
                    sh 'pip install -r dev-requirements.txt'
                    sh 'python project.py test CUDA'
                }
            }
        }
    }
}
