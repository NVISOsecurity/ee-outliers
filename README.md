
# ee-outliers 
Framework to easily detect outliers in Elasticsearch events.

*Developed in Python and fully dockerized!*

![version badge](https://img.shields.io/badge/version-0.2.11-blue "verion 0.2.11")
![tests badge](https://img.shields.io/badge/unit_tests-216-orange "216 unit tests")

## On this page

- [What is ee-outliers?](#what-is-ee-outliers)
- [Why ee-outliers?](#why-ee-outliers)
- [How it works](#how-it-works)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Other documentation
- [Installation](documentation/GETTING_STARTED.md)
- [Building detection use cases](documentation/CONFIG_OUTLIERS.md)
- [Whitelisting outliers](documentation/WHITELIST.md)
- [Notification system](documentation/NOTIFICATIONS.md)
- [Information for developers](documentation/DEVELOPMENT.md)
    - [UML](documentation/UML.md)
- [Screenshots](documentation/SCREENSHOTS.md)


## What is ee-outliers?
ee-outliers is a framework to detect statistical outliers in events stored 
in an Elasticsearch cluster. It uses easy to write user-defined configuration files 
to decide which & how events should be analysed for outliers.

The framework was developed for the purpose of detecting anomalies in 
security events, however it could just as well be used for the detection 
of outliers in other data.

The only thing you need is Docker and an Elasticsearch cluster and you are
ready to start your hunt for outlier events!

## Why ee-outliers?
Although we love Elasticsearch, its search language is still lacking support 
for complex queries that allow for advanced analysis and detection of outliers -
features we came to love while using other tools such as Splunk.
 
This framework tries to solve these limitations by allowing the user to write 
simple use cases that can help in spotting outliers in your data using statistical 
and machine learning models.

## How it works

The framework makes use of statistical models that are easily defined by the user 
in a configuration file. In case the models detect an outlier, the relevant 
Elasticsearch events are enriched with additional outlier fields. These fields 
can then be dashboarded and visualized using the tools of your choice 
(Kibana or Grafana for example).

The possibilities of the type of anomalies you can spot using ee-outliers 
is virtually limitless. A few examples of types of outliers we have detected
ourselves using ee-outliers during threat hunting activities include:

-	Detect beaconing (DNS, TLS, HTTP, etc.)
-	Detect geographical improbable activity
-	Detect obfuscated & suspicious command execution
-	Detect fileless malware execution
-	Detect malicious authentication events
-	Detect processes with suspicious outbound connectivity
-	Detect malicious persistence mechanisms (scheduled tasks, auto-runs, etc.)
-	…

Visit the page [Getting started](documentation/GETTING_STARTED.md) to get 
started with outlier detection in Elasticsearch yourself!

## Contact

ee-outliers is developed & maintained by NVISO Labs.

You can reach out to the developers of ee-outliers by creating an issue in github.  
For any other communication, you can reach out by sending us an e-mail 
at [research@nviso.be](mailto:research@nviso.be).

We write about our research on our blog: https://blog.nviso.be  
You can follow us on twitter: https://twitter.com/NVISO_Labs

<p align="left"> 
<img alt="NVISO Labs logo" src="documentation/images/NVISO%20Labs%20standard%20logo.png?raw=true" width="200"/><br/>
</p>

Thank you for using ee-outliers and we look forward to your feedback! 🐀

## License

ee-outliers is released under the GNU GENERAL PUBLIC LICENSE v3 (GPL-3).
[LICENSE](LICENSE)

## Acknowledgements
We are grateful for the support received by 
[INNOVIRIS](https://innoviris.brussels/) and the Brussels region in 
funding our Research & Development activities. 

<p align="right"><a href="documentation/GETTING_STARTED.md">Getting started &#8594;</a></p>
