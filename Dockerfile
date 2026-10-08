FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"
EXPOSE 8000
ENTRYPOINT ["leantrack"]
CMD ["--help"]
