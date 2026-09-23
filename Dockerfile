FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH ROOTCAUSE_DEMO=1 \
    ROOTCAUSE_APP_DB=/tmp/rootcause_app.sqlite3 PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .
EXPOSE 7860
CMD ["streamlit", "run", "frontend/app.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--server.headless", "true"]
