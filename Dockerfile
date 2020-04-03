FROM python:3.6
ARG timezone=Europe/Brussels

RUN apt-get update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata && \
    ln -snf /usr/share/zoneinfo/$timezone /etc/localtime && \
    echo "$timezone" > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Change locale to UTF-8
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8

ENV TZ=$timezone
RUN apt-get -y install sudo

RUN useradd -ms /bin/bash docker
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN adduser docker sudo
USER docker

# Install all Python requirements. Also see the remark above with all the RUN sudo pip commands.
USER root
ADD ./requirements.txt /app/requirements.txt
RUN sudo pip3 install -r /app/requirements.txt

ADD ./defaults /defaults
ADD ./use_cases /use_cases
ADD ./app/ /app

ADD ./VERSION /VERSION

# Let world write to unit test files so Jenkins and other tools can run and manipulate them
RUN chmod a+w -R /app/tests/unit_tests/files

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
