FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create the reporting directory
RUN mkdir -p /app/reporting

COPY . .

# Create start script using python -m for evidently
RUN echo '#!/bin/bash\npython project.py\ncd /app/reporting\npython -m http.server 8082' > /app/reporting/start.sh
RUN chmod +x /app/reporting/start.sh

CMD ["/app/reporting/start.sh"]