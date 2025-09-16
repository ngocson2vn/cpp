import tarfile
from pathlib import Path

def extract_tar_gz(archive_path: str, dest_dir: str):
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if dest_dir is None:
        dest_dir = archive_path.with_suffix('')  # remove last suffix (.gz) -> yields .tar; we'll strip again below
        # Ensure a clean, meaningful directory name
        if dest_dir.suffix == ".tar":
            dest_dir = dest_dir.with_suffix('')
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, mode="r:gz") as tar:
        try:
            tar.extractall(path=dest_dir)
            print(f"Successfully extracted tarball {archive_path} to {dest_dir}")
        except Exception as ex:
            print(f"Failed to extract tarball {archive_path}")
            raise ex

def create_tarball(src_dir):
    tar_path = f"{src_dir}.tar"
    src_dir = Path(src_dir)
    with tarfile.open(tar_path, mode="w") as tar:
        tar.add(src_dir, arcname=src_dir.name)
    print(f"Created tarball {tar_path}")


create_tarball("./data")
