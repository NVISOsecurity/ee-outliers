def latest_tag
def version_tag
def feature_version_tag
def push_images

pipeline {
	options{
        disableConcurrentBuilds()
        office365ConnectorWebhooks([[
            notifyBackToNormal: true,
            notifyFailure: true,
            notifyRepeatedFailure: true,
            notifySuccess: false,
            notifyUnstable: true,
            url: "${env.TEAMS_WEBHOOK}"
        ]])
	}
	
    agent { label 'docker' }

    stages {
        stage('Clone repository') {
            steps {
                checkout scm
            }
        }

        stage('Prepare version tags') {
            steps {
                script {
                    env.WORKSPACE = pwd()
                    def version = readFile "${env.WORKSPACE}/VERSION"
                    def feature_version = version.split("\\.")[0..1].join(".")
                    version = version.trim()

                    if(env.BRANCH_NAME == 'master') {
                        latest_tag = "latest"
                        version_tag = "${version}"
                        feature_version_tag = "${feature_version}"
                        push_images = "true"
                    }
                    else if(env.BRANCH_NAME == 'development') {
                        latest_tag = "devlatest"
                        version_tag = "${version}-dev"
                        feature_version_tag = "${feature_version}-dev"
                        push_images = "true"
                    }
                    else if(env.BRANCH_NAME =~ /^release-[\d\.]+$/) {
                        latest_tag = "rclatest"
                        version_tag = "${version}RC${env.BUILD_NUMBER}"
                        feature_version_tag = "${feature_version}RC${env.BUILD_NUMBER}"
                        push_images = "true"
                    }
                    else if(env.BRANCH_NAME =~ /^feature-.*$/) {
                        latest_tag = env.BRANCH_NAME
                        version_tag = env.BRANCH_NAME
                        feature_version_tag = env.BRANCH_NAME
                        push_images = "true"
                    } else {
                        latest_tag = env.BRANCH_NAME
                        version_tag = env.BRANCH_NAME
                        feature_version_tag = env.BRANCH_NAME
                        push_images = "false"
                    }
                }
            }
        }

        stage('Build docker image') {
            steps {
                script {
                    if(env.NO_CACHE == "1") {
                        app = docker.build("eagleeye/outliers:${latest_tag}", "--no-cache .")
                    } else {
                        app = docker.build("eagleeye/outliers:${latest_tag}")
                    }
                }
            }
        }

        stage('Test image') {
            steps {
                script {
                    app.inside {
                        sh 'python3 /app/outliers.py tests --config /defaults/outliers.conf --use-cases /app/tests/files/use_cases/*.conf'
                    }
                }
            }
            
        }

        stage('Sonarqube analysis') {
            steps {
                script{
                    env.VERSION = version_tag
                    def scannerHome = tool 'sonarscanner';
                    withSonarQubeEnv('Sonar') { 
                        sh "${scannerHome}/bin/sonar-scanner"
                    }
                }
            }
        }
        
        stage("Quality Gate") {
            steps {
                timeout(time: 30, unit: 'MINUTES') {
                    waitForQualityGate abortPipeline: true
                }
            }
        }
        
        stage('Push image') {
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
                    
                    if (push_images == 'true') {
                        docker.withRegistry("${env.DOCKER_REGISTRY_URL}", 'jenkins-nexus') {
                            app.push("${version_tag}")
                            app.push("${feature_version_tag}")
                            app.push("${latest_tag}")
                            
                        }
                    }
                    
                    if(env.BRANCH_NAME == 'master') {
                        withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
                            sh """
                                docker tag eagleeye/outliers:latest nvisobe/ee-outliers:${latest_tag};
                                docker tag eagleeye/outliers:latest nvisobe/ee-outliers:${version_tag};
                                docker tag eagleeye/outliers:latest nvisobe/ee-outliers:${feature_version_tag};
                                docker login --username=$USERNAME --password=$PASSWORD;
                                docker push nvisobe/ee-outliers:${latest_tag};
                                docker push nvisobe/ee-outliers:${version_tag};
                                docker push nvisobe/ee-outliers:${feature_version_tag};
                            """
                        }
                    }
                }
            }
        }
    }
}
