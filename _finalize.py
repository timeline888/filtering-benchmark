import subprocess, os
os.chdir(r"e:\Qoder项目\滤波基石系统设计")
# Check status
r = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
lines = [l for l in r.stdout.strip().split('\n') if l.strip()]
if lines:
    # Has changes, commit them
    subprocess.run(["git", "add", "-A"], capture_output=True)
    subprocess.run(["git", "commit", "-m", "清理: 移除临时脚本"], capture_output=True)
    push = subprocess.run(["git", "push", "origin", "master"], capture_output=True, text=True)
    with open(r"e:\git_cleanup_log.txt", "w") as f:
        f.write(f"Files committed: {lines}\n")
        f.write(f"Push stdout: {push.stdout}\n")
        f.write(f"Push stderr: {push.stderr}\n")
        f.write(f"Push rc: {push.returncode}\n")
else:
    with open(r"e:\git_cleanup_log.txt", "w") as f:
        f.write("Nothing to commit - working tree clean\n")
