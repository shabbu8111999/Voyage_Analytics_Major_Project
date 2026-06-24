# base image — Python 3.11 slim keeps the image small
FROM python:3.11-slim
 
# set working directory inside the container
WORKDIR /app
 
# copy requirements first so Docker caches this layer
# if requirements don't change, pip install is skipped on rebuild
COPY requirements.txt .
 
# install all packages
RUN pip install --no-cache-dir -r requirements.txt
 
# copy the rest of the project into the container
COPY . .
 
# tell Docker that the app runs on port 5000
EXPOSE 5000
 
# command to start the Flask server when container runs
CMD ["python", "api/app.py"]