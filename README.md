# cml_synthetic_demo
A Web application demo to discover actionable spatiotemporal descriptors of urban dynamics from MP Data.

Instructions to use the demo:

1. Download the synthetic dataset from https://www.kaggle.com/datasets/miguelgarcaosilva/synthetic-mp-data-in-lisbon
2. Extract its contents (SyntheticData folder) to the folder "cml_synthetic_demo/webapp-docker/devops/populate_db/"
3. Navigate to the cml_synthetic_demo/webapp-docker folder and run "docker-compose build" (This step can take several minutes when it's ran for the first time)
4. Run the command "docker-compose up"
5. Using a browser access the url "http://localhost:5000/home"

Requires the docker-compose command. (Refer to https://docs.docker.com/get-docker/ if you do not have docker and docker-compose installed).

Contact if you have any trouble testing the demo: mmgsilva@fc.ul.pt 