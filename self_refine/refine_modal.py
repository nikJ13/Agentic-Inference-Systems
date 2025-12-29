import pathlib
import modal

app = modal.App("self-refine-simple")
ROOT = pathlib.Path(__file__).resolve().parent
SRC_DIR = ROOT
RESULTS_VOLUME = modal.Volume.from_name("self-refine-results_2", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch==2.3.1",
        "transformers",
        "tokenizers",
        "datasets==2.20.0",
        "accelerate==0.31.0",
    )
    .env({"PYTHONPATH": "/root/self_refine/src", "HF_HOME": "/root/.cache/huggingface"})
    .add_local_dir(SRC_DIR, remote_path="/root/self_refine/src")
)

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 80,
    volumes={"/results": RESULTS_VOLUME},
)
def run_self_refine(args: list[str]):
    import pathlib
    import subprocess

    arg_list = list(args)
    output_name = None
    if "--output" in arg_list:
        out_idx = arg_list.index("--output")
        output_name = arg_list[out_idx + 1]
    else:
        output_name = "self_refine_output.jsonl"
        arg_list.extend(["--output", output_name])

    output_path = pathlib.Path("/results") / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arg_list[arg_list.index("--output") + 1] = str(output_path)

    cmd = ["python", "/root/self_refine/src/self_refine.py", *arg_list]
    subprocess.run(cmd, check=True)
    RESULTS_VOLUME.commit()

@app.local_entrypoint()
def main(
    handler: str = "graph",
    model_path: str = "Qwen/Qwen3-4B",
    output: str = "self_refine_output.jsonl",
    num_refine_steps: int = 3,
):
    args = [
        "--handler", handler,
        "--output", output,
        "--model-path", model_path,
        "--num-refine-steps", str(num_refine_steps),
        "--trust-remote-code"
    ]
    run_self_refine.remote(args)
