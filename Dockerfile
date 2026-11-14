FROM python:3.12-slim

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_THEME_BASE=dark \
    STREAMLIT_THEME_PRIMARY_COLOR="#E8590C" \
    STREAMLIT_THEME_BACKGROUND_COLOR="#0d1117" \
    STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR="#161b22" \
    STREAMLIT_THEME_TEXT_COLOR="#dde3ea" \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
# XSRF off: Spaces serve the app inside an iframe, which otherwise breaks file uploads
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableXsrfProtection=false"]
