FROM jupyter/base-notebook:latest

# Copy scripts or install requirements if necessary
# COPY requirements.txt /tmp/
# RUN pip install -r /tmp/requirements.txt

# Set the working directory
WORKDIR /scripts

# Start Jupyter Notebook
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.allow_origin='*'"]
