import modal
import subprocess
import os
import argparse

app = modal.App("vggt")
# image = modal.Image.from_gcp_artifact_registry(
#     "gcr.io/tour-project-442218/vggsfm",
#     secret=modal.Secret.from_name(
#         "gcp-credentials",
#         required_keys=["SERVICE_ACCOUNT_JSON"],
#     ),
# ).add_local_dir(".", "/root/")

image = modal.Image.from_dockerfile("Dockerfile").add_local_dir(".", "/vggt/")

@app.function(gpu="A100-80GB", image=image, secrets=[modal.Secret.from_name("gcp-credentials")], timeout=3600)
def train_vggt(dataset: str | None = None):
    print("Current working directory:", os.getcwd()) # Note that this is /root and the modal file is copied in here..

    # Make the shell script executable
    os.chmod("run_training.sh", 0o755)

    try:
        # Run the shell script with dataset argument
        subprocess.run(["./run_training.sh", dataset or ""], check=True)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        raise

@app.local_entrypoint()
def main(dataset: str | None = None):
    train_vggt.remote(dataset)
