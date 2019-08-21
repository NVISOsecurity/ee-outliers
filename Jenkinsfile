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

        stage('Sonarqube analysis') {
            when{
                branch "development"
            }
            steps {
                script{
                    withCredentials([string(credentialsId: 'sonar-login-key', variable: 'LOGIN')]) {
                        sh '''
                            /opt/sonar-scanner -Dsonar.login=$LOGIN
                        '''
                    }
                }
            }
        }

        stage('Push image') {
            steps {
                script {
                    env.WORKSPACE = pwd()
                    def version = readFile "${env.WORKSPACE}/VERSION"
                    def full_version = version.trim()
                    def feature_version = full_version.split("\\.")[1..2].join(".")
                    def latest_tag = ""
                    if(env.BRANCH_NAME == 'master') {
                        latest_tag = "latest"
                    } else if(env.BRANCH_NAME == 'development') {
                        latest_tag = "devlatest"
                        full_version = "${full_version}-dev"
                        feature_version = "${feature_version}-dev"
                    }
                    if (latest_tag != "") {
                        docker.withRegistry('https://localhost:1234/', 'jenkins-nexus') {
                            app.push("${full_version}")
                            app.push("${feature_version}")
                            app.push("${latest_tag}")
                        }
                    }
                }
            }
        }
        stage('Official release') {
            steps {
                script {
                    if (env.BRANCH_NAME == 'master') {
                        sshagent (credentials: ['GithubSSHKey']) {
                            sh '''
                                if ! git tag --list $(cat VERSION); then
                                    git tag $(cat VERSION)
                                    git push origin --tags
                                fi
                            '''
                        }
                    }
                }
            }
        }
    }

    post {
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

