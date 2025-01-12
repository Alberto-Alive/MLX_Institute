# Start off with an image that has python version 3.11 installed
FROM python:3.11

# Create a directory and move into it
WORKDIR /code


# Copy the requirements in first, so that when we rebuild the image, it
# can use the same layer (essentially the cache), and skip this part
COPY ./requirements.txt /code/requirements.txt

# Install all the requirements. Will also be skipped if not changes
# have been made
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy inside the rest of our app that 
COPY . ./app

EXPOSE 8000

# Define the command to run your script
# CMD ["python", "model_hk_titles.py"]
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
