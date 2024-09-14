FROM python:3.12.3

WORKDIR /Yt_dashboard

COPY  requirements.txt  ./requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8501

COPY . /Yt_dashboard

ENTRYPOINT ["streamlit", "run"]

CMD ["Yt_dashboard.py"]


