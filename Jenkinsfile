pipeline {
    agent {
        label 'docker'
    }

    stages {
        stage('Clone repository') {  
            steps {
                script {
                    checkout scm
                }   
            }
        }

        stage('Build docker image') {
            steps {
                script {
                    if(env.NO_CACHE == "1") {
                        app = docker.build("eagleeye/outliers", "--no-cache .")
                    } else {
                        app = docker.build("eagleeye/outliers")
                    }
                }
            }
        }


        stage('Test image') {
            steps {
                script {
                    app.inside {
                        sh 'python3 /app/outliers.py tests --config /defaults/outliers.conf'
                    }
                }
            }
            
        }

        stage('Push image') {
            steps {
                script {
                    env.WORKSPACE = pwd()
                    def version = readFile "${env.WORKSPACE}/VERSION"
                    version = version.trim()
                    def latest_tag = ""
                    if(env.BRANCH_NAME == 'master') {
                        latest_tag = "latest"
                    } else if(env.BRANCH_NAME == 'development') {
                        latest_tag = "devlatest"
                    }
                    if (latest_tag != "") {
                        docker.withRegistry('https://localhost/', 'docker-registry-basic') {
                            app.push("${version}r${env.BUILD_NUMBER}-${env.BRANCH_NAME}")
                            app.push("${latest_tag}")
                        }
                    }
                }
            }
        }
    }

    post {
        success {
            slackSend color: 'good', message: "Build Succeeded! - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
        }
        failure {
            slackSend color: 'danger', message: "Build Failed! - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
        }
        always {
            script {
                sh 'docker run -v $WORKSPACE:/workspace --rm alpine/flake8 --exit-zero --ignore=E501 --output-file=/workspace/flake8.xml /workspace/app'
                def flake8 = scanForIssues filters: [], tool: flake8(pattern: 'flake8.xml')
                archiveArtifacts 'flake8.xml'
                publishIssues issues: [flake8]
            }
        }
    }
    
}

