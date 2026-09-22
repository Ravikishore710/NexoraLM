import argparse
import json
import os
import subprocess
import sys
import time

def create_kernel_metadata(kernel_dir, kernel_slug, title, code_file, is_gpu=True):
    username = os.environ.get("KAGGLE_USERNAME", "raviikishore07")
    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if is_gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }
    meta_path = os.path.join(kernel_dir, "kernel-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return meta_path

def push_kernel(kernel_dir):
    cmd = ["kaggle", "kernels", "push", "-p", kernel_dir]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error pushing kernel: {res.stderr}")
        return False
    print(res.stdout.strip())
    return True

def check_status(kernel_slug):
    username = os.environ.get("KAGGLE_USERNAME", "raviikishore07")
    kernel_id = f"{username}/{kernel_slug}"
    cmd = ["kaggle", "kernels", "status", kernel_id]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error checking status: {res.stderr}")
        return None
    status = res.stdout.strip()
    print(status)
    return status

def download_output(kernel_slug, output_dir):
    username = os.environ.get("KAGGLE_USERNAME", "raviikishore07")
    kernel_id = f"{username}/{kernel_slug}"
    os.makedirs(output_dir, exist_ok=True)
    cmd = ["kaggle", "kernels", "output", kernel_id, "-p", output_dir]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error downloading output: {res.stderr}")
        return False
    print(f"Downloaded output to {output_dir}")
    return True

def poll_kernel(kernel_slug, poll_interval=30):
    print(f"Monitoring kernel {kernel_slug}...")
    while True:
        status = check_status(kernel_slug)
        if not status:
            break
        if "complete" in status.lower():
            print("Kernel execution completed successfully.")
            return True
        if "error" in status.lower() or "failed" in status.lower():
            print("Kernel execution failed.")
            return False
        time.sleep(poll_interval)

def main():
    parser = argparse.ArgumentParser(description="Kaggle kernel automation runner")
    subparsers = parser.add_subparsers(dest="command")

    push_p = subparsers.add_parser("push")
    push_p.add_argument("--dir", required=True, help="Directory containing code and metadata")
    push_p.add_argument("--slug", required=True, help="Kernel slug")
    push_p.add_argument("--title", required=True, help="Kernel title")
    push_p.add_argument("--script", required=True, help="Entry script name inside dir")
    push_p.add_argument("--gpu", action="store_true", default=True, help="Enable GPU")

    status_p = subparsers.add_parser("status")
    status_p.add_argument("--slug", required=True, help="Kernel slug")

    output_p = subparsers.add_parser("output")
    output_p.add_argument("--slug", required=True, help="Kernel slug")
    output_p.add_argument("--out", default="outputs", help="Output directory")

    poll_p = subparsers.add_parser("poll")
    poll_p.add_argument("--slug", required=True, help="Kernel slug")
    poll_p.add_argument("--interval", type=int, default=30, help="Polling interval in seconds")

    args = parser.parse_args()

    if args.command == "push":
        create_kernel_metadata(args.dir, args.slug, args.title, args.script, is_gpu=args.gpu)
        push_kernel(args.dir)
    elif args.command == "status":
        check_status(args.slug)
    elif args.command == "output":
        download_output(args.slug, args.out)
    elif args.command == "poll":
        poll_kernel(args.slug, args.interval)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
