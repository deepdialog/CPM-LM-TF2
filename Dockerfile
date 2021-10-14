FROM python:3.8
RUN useradd -ms /bin/bash qhduan
USER qhduan
WORKDIR /home/qhduan
ENV PATH="/home/qhduan/.local/bin:${PATH}"
COPY --chown=qhduan:qhduan ./requirements.txt .
RUN pip install --upgrade --user pip -i https://mirrors.aliyun.com/pypi/simple
RUN pip install --user -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
COPY --chown=qhduan:qhduan ./onnx_q ./onnx_q
COPY --chown=qhduan:qhduan ./onnx_kv_q ./onnx_kv_q
COPY --chown=qhduan:qhduan ./tokenizer ./tokenizer
COPY --chown=qhduan:qhduan ./main.py .
COPY --chown=qhduan:qhduan ./infer.py .
COPY --chown=qhduan:qhduan ./scripts ./scripts
COPY --chown=qhduan:qhduan ./gpt2_tokenizer.py .
EXPOSE 8000
HEALTHCHECK CMD curl --fail http://localhost:8000 || exit 1
ENTRYPOINT ["/home/qhduan/scripts/run.sh"]

