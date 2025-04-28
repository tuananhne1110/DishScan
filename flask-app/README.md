# Model Deployment as an API using Flask

### Overview

- The Flask app, together with the model is packed inside Docker, all the necessy steps are included in the Dockerfile, make the app easy to deploy.

- The Flask app is started at port 5000 by default.

### Instructions

- All the necessary files are provided, to build the image, run the following command in this `/flask-app` directory:

```sh
docker build -t <your_image_name> .
```

- To start the app, use:

```sh
docker run -p 5000:5000 <your_image_name>
```

- Make sure to publish the port via the `-p` flag, the Flask app is now initialized and running at port `5000`