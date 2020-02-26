pipeline {

    options {
	    office365ConnectorWebhooks([[
		notifyBackToNormal: true,
		notifyFailure: true,
		notifyRepeatedFailure: true,
		notifySuccess: false,
		notifyUnstable: true,
		url: "${env.TEAMS_WEBHOOK}"
	    ]])
    }
	
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
                	def scannerHome = tool 'sonarscanner';
				    withSonarQubeEnv('Sonar') { 
				      sh "${scannerHome}/bin/sonar-scanner"
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
                    def feature_version = full_version.split("\\.")[0..1].join(".")
                    docker.withRegistry("${env.DOCKER_REGISTRY_URL}", 'jenkins-nexus') {
                        if(env.BRANCH_NAME == 'master') {
                            app.push("${full_version}")
                            app.push("${feature_version}")
                            app.push("latest")
                        } else if(env.BRANCH_NAME == 'development') {
                            app.push("devlatest")
                        } else if (env.BRANCH_NAME == 'hotfix-bulk-flush') {
                            app.push("${full_version}")
                            app.push("${feature_version}")
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
                                git tag $(cat VERSION)
                                git push origin --tags
                            '''
                        }
                    }
                }
            }
        }
    }
    
}

