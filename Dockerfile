# 
FROM python:3.8.16

# 
WORKDIR /code

COPY . .

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# 
RUN pip install --no-cache-dir --upgrade -r newreq.txt

EXPOSE 8000
# 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
