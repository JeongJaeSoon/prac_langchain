FROM python:3.11.5-slim AS builder

RUN pip install --no-cache-dir poetry==1.6.1

WORKDIR /opt/builder
COPY ./pyproject.toml ./poetry.lock /opt/builder/

RUN poetry export --output requirements.txt --without-hashes

FROM python:3.11.5-slim

RUN apt-get update -y && \
    apt-get install -y fonts-ipafont-gothic fonts-ipafont-mincho libglib2.0-0 libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libdbus-1-3 libatspi2.0-0 libx11-6 libxcomposite1 libxdamage1 libxext6 libxfixes3 libxrandr2 libgbm1 libxcb1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "nonroot:x:65532:65532::/home/nonroot:/usr/sbin/nologin" >> /etc/passwd
RUN echo "nonroot:x:65532:" >> /etc/group
RUN mkdir /home/nonroot && chown nonroot:nonroot /home/nonroot

USER 65532

WORKDIR /home/nonroot

COPY --from=builder /opt/builder/requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /opt/requirements.txt

ENV PATH="/home/nonroot/.local/bin:${PATH}"

COPY . /opt/.

WORKDIR /opt/.

ENV PYTHONPATH /opt/.

EXPOSE 8080
ENTRYPOINT [ "uvicorn", "app:fastapi_app", "--host", "0.0.0.0", "--port", "8080" ]
