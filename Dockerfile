FROM python:3.10

COPY pyproject.toml /app/pyproject.toml
RUN curl https://get.modular.com | \
  MODULAR_AUTH=mut_793643b228a947269687379f7ad596d4 \
  sh -
RUN modular install mojo
RUN echo 'export MODULAR_HOME="/root/.modular"' >> ~/.bashrc
RUN echo 'export PATH="/root/.modular/pkg/packages.modular.com_mojo/bin:$PATH"' >> ~/.bashrc
RUN source ~/.bashrc
RUN pip install -e /app

COPY . /app

ENTRYPOINT ["mojo", "run", "/app/main.mojo"]
