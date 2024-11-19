# Building a transformer from scratch

This is a repo to learn how to build a transformer from scratch. Planning to build out further infrastructure afterwards...
Might do some visualization of weights in the future...

### Setting up:
- Create a .env file in the base directory of the project
- Create a wandb account and enter: WANDB_API_KEY = ["YOUR_API_KEY"]

To create the environment:
make all
conda activate transformer_env



### Running on cloud instances:
I'm using a single t4 GCP Ubuntu instance
gcloud compute ssh [INSTANCE_NAME] --zone=[ZONE]
