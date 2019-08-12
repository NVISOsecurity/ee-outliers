# Notifications

A system of notification have been implemented in ee-outlier. This system allow you to receive an email when an Outlier have been detected.  Go to the configuration to setup the notification system.

**email_notifier**
Set it to 1 to enable the notification system.

**notification_email**
Email where the information need to be send

**smtp_user, smtp_pass, smtp_server, smtp_port**
SMTP information to connect to mail server


These information are not sufficient to receive notifications.  Indeed, the notification system is not global. You need to enable it for each model you want to receive an alert.
To do that you need to set `should_notify` to `1` for the models that you would like enable.

