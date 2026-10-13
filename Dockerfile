FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg PORT=8000
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 libportaudio2 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt
COPY app.py model.py lift_counter.py serve.py download_model.py ./
COPY static ./static
RUN python download_model.py
EXPOSE 8000
CMD ["python", "serve.py"]
