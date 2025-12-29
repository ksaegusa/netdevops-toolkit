FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir uv
RUN uv pip compile pyproject.toml -o /tmp/requirements.txt \
    && uv pip install --system --no-cache-dir -r /tmp/requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
