FROM node:16 as frontend

# set working directory
WORKDIR /app

# add `/app/node_modules/.bin` to $PATH
ENV PATH /app/node_modules/.bin:$PATH

# install app dependencies
COPY ./autochemplete/static/js ./
RUN npm install --silent

RUN npm run build

FROM python:3.10 as requirements-stage

WORKDIR /tmp

RUN pip install poetry

COPY ./pyproject.toml ./poetry.lock* /tmp/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.10

WORKDIR /app

COPY --from=requirements-stage /tmp/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./ /app
COPY --from=frontend /app /app/autochemplete/static/js

RUN pip install . 

CMD ["gunicorn", "autochemplete.main:app", "--workers", "2", "--bind", "0.0.0.0:80", "--timeout", "300", "--log-level=debug"]
