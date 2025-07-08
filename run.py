import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import modal

MODAL_SECRETS = [modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("github-token")]
MODAL_VOLUMES = {
    "/root/data": modal.Volume.from_name("data", create_if_missing=True),
    "/root/output": modal.Volume.from_name("output", create_if_missing=True),
}

def dummy_function():
    # Testing whether this could get models downloaded and cuda things prebuilt
    # but needs to be placed into a python function unfortunately so that modal can properly
    # run it with `run_function` and attach a volume
    print("Running dummy function")
    subprocess.run("python demo_colmap.py --scene_dir=~/data/examples/kitchen --query_frame_num=1 --max_query_pts=100", shell=True, cwd=".")


app = modal.App("vggt", image=modal.Image.from_dockerfile(Path(__file__).parent / "Dockerfile")
    # GCloud
    .add_local_file(Path.home() / "gcs-tour-project-service-account-key.json", "/root/gcs-tour-project-service-account-key.json", copy=True)
    .run_commands(
        "gcloud auth activate-service-account --key-file=/root/gcs-tour-project-service-account-key.json",
        "gcloud config set project tour-project-442218",
        "gcloud storage ls"
    )
    .env({"GOOGLE_APPLICATION_CREDENTIALS": "/root/gcs-tour-project-service-account-key.json"})
    .run_commands("gcloud storage ls")
    # SSH server
    .apt_install("openssh-server")
    .run_commands(
        "mkdir -p /run/sshd" #, "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config", "echo 'root: ' | chpasswd" #TODO: uncomment this if the key approach doesn't work
    )
    .add_local_file(Path.home() / ".ssh/id_rsa.pub", "/root/.ssh/authorized_keys", copy=True)
    # Add Conda (for some reason necessary for ssh-based code running)
    .run_commands("conda init bash && echo 'conda activate base' >> ~/.bashrc")
    # Fix Git
    .run_commands("git config --global pull.rebase true")
    .run_commands("git config --global user.name 'Nikita Demir'")
    .run_commands("git config --global user.email 'nikitde1@gmail.com'")
    # Set CUDA Architecture (depends on the GPU)
    .env({"TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.9"})
    # Add Our Code and Install EDGS
    .workdir("/root/workspace")
    # Clone EDGS repository into current directory
    .run_commands("git clone https://github.com/N-Demir/vggt.git . --recursive")
    # Install
    .run_commands("pip install -r requirements.txt")
    .run_commands("pip install -r requirements_demo.txt")
    .run_commands("pip install -e '.[demo]'")
    .run_commands("apt install rsync -y") # for copying images
    # # Post install, try actually running a demo example to prebuild/download things
    .run_commands("git pull")
    .run_function(dummy_function, secrets=MODAL_SECRETS, volumes=MODAL_VOLUMES, gpu="A100-80GB")
    # Get the latest code
    .run_commands("git pull", force_build=True)
)


LOCAL_PORT = 9090


def wait_for_port(host, port, q):
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 22), timeout=30.0):
                break
        except OSError as exc:
            time.sleep(0.01)
            if time.monotonic() - start_time >= 30.0:
                raise TimeoutError("Waited too long for port 22 to accept connections") from exc
        q.put((host, port))


@app.function(
    timeout=3600 * 24,
    gpu="T4",
    secrets=MODAL_SECRETS,
    volumes=MODAL_VOLUMES
)
def run_server(q):
    with modal.forward(22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        threading.Thread(target=wait_for_port, args=(host, port, q)).start()

        # Added these commands to get the env variables that docker loads in through ENV to show up in my ssh
        import os
        import shlex
        from pathlib import Path

        output_file = Path.home() / "env_variables.sh"

        with open(output_file, "w") as f:
            for key, value in os.environ.items():
                escaped_value = shlex.quote(value)
                f.write(f'export {key}={escaped_value}\n')
        subprocess.run("echo 'source ~/env_variables.sh' >> ~/.bashrc", shell=True)

        subprocess.run(["/usr/sbin/sshd", "-D"])  # TODO: I don't know why I need to start this here


@app.function(
    timeout=3600 * 24,
    gpu="A100-80GB",
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("github-token")],
    volumes={
             "/root/data": modal.Volume.from_name("data", create_if_missing=True),
             "/root/output": modal.Volume.from_name("output", create_if_missing=True),
             "/root/ever_training": modal.Volume.from_name("ever-training", create_if_missing=True)}
)
def run_shell_script(shell_file_path: str):
    """Run a shell script on the remote Modal instance."""
    # Run the shell script
    print(f"Running shell script: {shell_file_path}")
    subprocess.run("bash " + shell_file_path, 
                  shell=True, 
                  cwd=".")


@app.local_entrypoint()
def main(server: bool = False, shell_file: str | None = None):   
    if server:
        import sshtunnel

        with modal.Queue.ephemeral() as q:
            run_server.spawn(q)
            host, port = q.get()
            print(f"SSH server running at {host}:{port}")

            ssh_tunnel = sshtunnel.SSHTunnelForwarder(
                (host, port),
                ssh_username="root",
                ssh_password=" ",
                remote_bind_address=("127.0.0.1", 22),
                local_bind_address=("127.0.0.1", LOCAL_PORT),
                allow_agent=False,
            )

            try:
                ssh_tunnel.start()
                print(f"SSH tunnel forwarded to localhost:{ssh_tunnel.local_bind_port}")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down SSH tunnel...")
            finally:
                ssh_tunnel.stop()

    if shell_file:
        # Run the shell script on the remote instance
        print(f"Running shell script: {shell_file}")
        run_shell_script.remote(shell_file)