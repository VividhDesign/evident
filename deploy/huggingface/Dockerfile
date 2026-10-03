# Hugging Face Space (Docker SDK): live demo of Evident (RAG) running on Strata (C++ vector DB).
# The code is pulled from GitHub at build time; the Space repo itself only carries this file,
# the Space card and precomputed data (BEIR corpora + document embeddings).
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake ninja-build git \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PIP_NO_CACHE_DIR=1
WORKDIR /home/user

# CPU-only PyTorch: the default CUDA build is ~2 GB larger and unused on CPU hardware.
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu

# Re-download the repos whenever their main branch moves (busts Docker's layer cache).
ADD --chown=user https://api.github.com/repos/VividhDesign/strata/git/refs/heads/main /tmp/strata-ref.json
# Portable AVX2 build: the machine that builds this image may not be the one that runs it.
RUN pip install --user --config-settings=cmake.define.STRATA_NATIVE=OFF \
    "strata-vdb @ git+https://github.com/VividhDesign/strata"

ADD --chown=user https://api.github.com/repos/VividhDesign/evident/git/refs/heads/main /tmp/evident-ref.json
RUN git clone --depth 1 https://github.com/VividhDesign/evident.git \
 && pip install --user -e "./evident[app]"

# Bake the models into the image so the first visitor doesn't wait for downloads.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user data /home/user/data

ENV EVIDENT_DATA=/home/user/data \
    EVIDENT_HOSTED=1 \
    EVIDENT_LLM_PROVIDER=groq \
    EVIDENT_LLM_MODEL=llama-3.1-8b-instant \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /home/user/evident
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
