# ee-outliers
Framework to easily detect outliers in Elasticsearch events.

*Developed in Python and fully dockerized!*

![version badge](https://img.shields.io/badge/version-0.2.11-blue "verion 0.2.11")
![tests badge](https://img.shields.io/badge/unit_tests-216-orange "216 unit tests")

## Table of contents
- [What is ee-outliers?](#what-is)
- [Why ee-outliers?](#why)
- [How it works](#how)
- [Screenshots](#screenshots)
- [License](LICENSE)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
- [Getting started](documentation/INSTALL.md)
- [Building detection use cases](documentation/CONFIG_OUTLIERS.md)
- [Whitelisting outliers](documentation/WHITELIST.md)
- [Notification system](documentation/NOTIFICATIONS.md)
- [Information for developers](documentation/DEVELOPMENT.md)
    - [UML](documentation/UML.md)



## [What is ee-outliers?](what-is)
ee-outliers is a framework to detect statistical outliers in events stored 
in an Elasticsearch cluster. It uses easy to write user-defined configuration files 
to decide which & how events should be analysed for outliers.

The framework was developed for the purpose of detecting anomalies in 
security events, however it could just as well be used for the detection 
of outliers in other data.

The only thing you need is Docker and an Elasticsearch cluster and you are
ready to start your hunt for outlier events!

## [Why ee-outliers?](why)
Although we love Elasticsearch, its search language is still lacking support 
for complex queries that allow for advanced analysis and detection of outliers -
features we came to love while using other tools such as Splunk.
 
This framework tries to solve these limitations by allowing the user to write simple use cases
that can help in spotting outliers in your data using statistical and machine 
learning models.

## [how it works](how)

The framework makes use of statistical models that are easily defined by the user in a configuration file. In case the 
models detect an outlier, the relevant Elasticsearch events are enriched with additional outlier fields. These fields 
can then be dashboarded and visualized using the tools of your choice (Kibana or Grafana for example).

The possibilities of the type of anomalies you can spot using ee-outliers is virtually limitless. A few examples of 
types of outliers we have detected ourselves using ee-outliers during threat hunting activities include:

-	Detect beaconing (DNS, TLS, HTTP, etc.)
-	Detect geographical improbable activity
-	Detect obfuscated & suspicious command execution
-	Detect fileless malware execution
-	Detect malicious authentication events
-	Detect processes with suspicious outbound connectivity
-	Detect malicious persistence mechanisms (scheduled tasks, auto-runs, etc.)
-	…

Checkout the screenshots at the end of this readme for a few examples.
Visit the page [Getting started](documentation/INSTALL.md) to get started with outlier 
detection in Elasticsearch yourself!

## [Contact](contact)

You can reach out to the developers of ee-outliers by creating an issue in github.
For any other communication, you can reach out by sending us an e-mail at [research@nviso.be](mailto:research@nviso.be).

Thank you for using ee-outliers and we look forward to your feedback! 🐀

## [Acknowledgements](acknowledgements)
ee-outliers is developed by NVISO Labs (https://blog.nviso.be - https://twitter.com/NVISO_Labs)

We are grateful for the support received by [INNOVIRIS](https://innoviris.brussels/) and the Brussels region in 
funding our Research & Development activities. 


<p align="right"><a href="documentation/INSTALL.md">Getting started &#8594;</a></p>

## [Screenshots](screenshots)
<p align="center"> 
<img alt="Detecting beaconing TLS connections using ee-outliers" src="https://forever.daanraman.com/screenshots/Beaconing%20detection.png?raw=true" width="650"/><br/>
<i>Detecting beaconing TLS connections using ee-outliers</i>
</p>
<br/><br/>  
<p align="center"> 
<img alt="Configured use case to detect beaconing TLS connections" src="https://forever.daanraman.com/screenshots/Configuration%20use%20case.png?raw=true" width="450"/><br/>
<i>Configured use case to detect beaconing TLS connections</i>
</p>
<br/><br/>
<p align="center"> 
<img alt="Detected outlier events are enriched with new fields in Elasticsearch" src="https://forever.daanraman.com/screenshots/Enriched%20outlier%20event%202.png?raw=true" width="650"/><br/>
<i>Detected outlier events are enriched with new fields in Elasticsearch</i>
</p>
