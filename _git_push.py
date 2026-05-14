import subprocess, os

os.chdir(r"e:\Qoder项目\滤波基石系统设计")

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"CMD: {cmd}")
    print(f"STDOUT: {result.stdout}")
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    print(f"RC: {result.returncode}")
    print("-" * 40)

run("git --version")
run("git log --oneline -3")
run("git status --short")
run("git push origin master")
