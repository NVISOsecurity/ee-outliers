# ee-outliers

```
                               __  ___
  ___  ___        ____  __  __/ /_/ (_)__  __________
 / _ \/ _ \______/ __ \/ / / / __/ / / _ \/ ___/ ___/
/  __/  __/_____/ /_/ / /_/ / /_/ / /  __/ /  (__  )
\___/\___/      \____/\__,_/\__/_/_/\___/_/  /____/

Open-source framework to detect outliers in Elasticsearch events
Developed by NVISO Labs (https://blog.nviso.be - https://twitter.com/NVISO_Labs)
```

**Table of contents**
- [Introduction](#introduction)
- [Getting started](documentation/INSTALL.md)
- [Building detection use cases](documentation/CONFIG_OUTLIERS.md)
- [Whitelisting outliers](documentation/WHITELIST.md)
- [Information fo developers](documentation/DEVELOPMENT.md)
- [Screenshots](#screenshots)
- [License](LICENSE)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Introduction

ee-outliers is a framework to detect outliers in events stored in an Elasticsearch cluster.
The framework was developed for the purpose of detecting anomalies in security events, however it could just as well be used for the detection of outliers in other types of data.


The framework makes use of statistical models that are easily defined by the user in a configuration file. In case the models detect an outlier, the relevant Elasticsearch events are enriched with additional outlier fields. These fields can then be dashboarded and visualized using the tools of your choice (Kibana or Grafana for example).

The possibilities of the type of anomalies you can spot using ee-outliers is virtually limitless. A few examples of types of outliers we have detected ourselves using ee-outliers during threat hunting activities include:

-	Detect beaconing (DNS, TLS, HTTP, etc.)
-	Detect geographical improbable activity
-	Detect obfuscated & suspicious command execution
-	Detect fileless malware execution
-	Detect malicious authentication events
-	Detect processes with suspicious outbound connectivity
-	Detect malicious persistence mechanisms (scheduled tasks, auto-runs, etc.)
-	…

Checkout the screenshots at the end of this readme for a few examples.
Continue reading if you would like to get started with outlier detection in Elasticsearch yourself!

#### Core features
- Create your own custom outlier detection use cases specifically for your own needs
- Send automatic e-mail notifications in case one of your outlier use cases hit
- Automatic tagging of asset fields to quickly spot the most interesting assets to investigate
- Fine-grained control over which historical events are checked for outliers
- ...and much more!

## Screenshots

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


## Contact

You can reach out to the developers of ee-outliers by creating an issue in github.
For any other communication, you can reach out by sending us an e-mail at research@nviso.be.

Thank you for using ee-outliers and we look forward to your feedback! 🐀

## Acknowledgements

We are grateful for the support received by [INNOVIRIS](https://innoviris.brussels/) and the Brussels region in funding our Research & Development activities. 


<p align="right"><a href="documentation/INSTALL.md">Getting started &#8594;</a></p>
