# ── Stage 1: build SWI-Prolog + janus_swi from source ──────────────────────
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV SWIPL_VERSION=V9.2.9

RUN sed -i \
    -e 's|http://archive.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' \
    -e 's|http://security.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' \
    /etc/apt/sources.list && \
    printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\nAcquire::http::Pipeline-Depth "0";\n' \
    > /etc/apt/apt.conf.d/80-retries

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git pkg-config ca-certificates \
    libssl-dev libgmp-dev libarchive-dev zlib1g-dev \
    libpcre2-dev libedit-dev libreadline-dev libossp-uuid-dev \
    python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${SWIPL_VERSION} \
    https://github.com/SWI-Prolog/swipl-devel.git /tmp/swipl-devel && \
    git -C /tmp/swipl-devel submodule update --init --depth 1 && \
    cmake -S /tmp/swipl-devel -B /tmp/swipl-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr && \
    cmake --build /tmp/swipl-build && \
    cmake --install /tmp/swipl-build && \
    swipl --version && \
    rm -rf /tmp/swipl-devel /tmp/swipl-build

RUN git clone https://github.com/SWI-Prolog/packages-swipy /tmp/packages-swipy && \
    cd /tmp/packages-swipy && \
    pip3 install .

# ── Stage 2: runtime image ──────────────────────────────────────────────────
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PETTA_COMMIT=e1490899cefc67c128d5311ff4861f9997674957
ENV PETTACHAINER_COMMIT=6b88df7c903705a38205709151cdd7549fd8d1b0

RUN sed -i \
    -e 's|http://archive.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' \
    -e 's|http://security.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' \
    /etc/apt/sources.list && \
    printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\nAcquire::http::Pipeline-Depth "0";\n' \
    > /etc/apt/apt.conf.d/80-retries

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgmp10 libarchive13 libpcre2-8-0 libedit2 libreadline8 \
    libossp-uuid16 libssl3 zlib1g \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/bin/swipl /usr/bin/swipl
COPY --from=builder /usr/bin/swipl-ld /usr/bin/swipl-ld
COPY --from=builder /usr/lib/swipl /usr/lib/swipl

RUN python3 -m pip install --upgrade pip setuptools wheel

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Clone PeTTa and PeTTaChainer
WORKDIR /deps
RUN git config --global http.version HTTP/1.1 && \
    git config --global http.postBuffer 524288000 && \
    for attempt in 1 2 3; do \
        git clone --filter=blob:none https://github.com/trueagi-io/PeTTa.git && break; \
        rm -rf /deps/PeTTa; \
        sleep 5; \
    done && \
    git -C /deps/PeTTa checkout ${PETTA_COMMIT} && \
    for attempt in 1 2 3; do \
        git clone --filter=blob:none https://github.com/rTreutlein/PeTTaChainer.git && break; \
        rm -rf /deps/PeTTaChainer; \
        sleep 5; \
    done && \
    git -C /deps/PeTTaChainer checkout ${PETTACHAINER_COMMIT}

# Install PeTTa — strip the janus-swi pip dep, then point PYTHONPATH
# at the actual source dir so `import petta` resolves correctly
RUN cd /deps/PeTTa && \
    sed -i "/'janus-swi'/d" setup.py && \
    pip3 install .

ENV PYTHONPATH=/deps/PeTTa/python:$PYTHONPATH

# Install PeTTaChainer
RUN cd /deps/PeTTaChainer && \
    pip3 install -e .

# Install pln-rag app dependencies
WORKDIR /app
ARG REQUIREMENTS_FILE=requirements.txt
ARG INSTALL_COREF=false
COPY requirements*.txt ./
RUN rm -rf /usr/lib/python3/dist-packages/blinker*
RUN pip3 install --default-timeout=1000 -r ${REQUIREMENTS_FILE}
RUN if [ "$INSTALL_COREF" = "true" ]; then \
        pip3 install --default-timeout=1000 -r requirements-coref.txt; \
    fi

COPY . .

VOLUME ["/app/data"]

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
