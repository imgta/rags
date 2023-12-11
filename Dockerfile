FROM python:3.11-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install -U pip setuptools wheel
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver --use-feature=fast-deps -r requirements.txt

COPY main.py ./
COPY .streamlit ./.streamlit

EXPOSE 8501

# Run streamlit when the container launches
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
