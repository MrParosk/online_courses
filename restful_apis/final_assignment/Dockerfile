FROM ubuntu:16.04

RUN apt-get -y upgrade
RUN apt-get -y install python3.6
RUN apt-get -y install python3-pip
RUN apt-get -y install git
RUN git clone https://github.com/parlstrand/online_courses/
WORKDIR "online_courses/restful_apis/final_assignment"
RUN pip3 install -r requirements.txt

EXPOSE 5000
ENTRYPOINT [ "python3" ]
CMD [ "run.py" ]