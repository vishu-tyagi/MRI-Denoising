ARG IMAGE
FROM ${IMAGE}

COPY . .

RUN pip install -e src/complex-torch

RUN chmod +x ./entrypoint.sh

ENTRYPOINT [ "./entrypoint.sh" ]