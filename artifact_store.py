"""Verify uploaded artifact sizes and LFS SHA256 / Git blob SHA1 hashes."""
import hashlib
import os
from huggingface_hub import HfApi

def upload_verified(folder, destination):
    api = HfApi(token=os.environ['HF_TOKEN'])
    repo = os.environ['HF_REPO_ID']
    if not api.model_info(repo).private:
        raise RuntimeError('Refusing artifact upload to non-private repo')
    api.upload_folder(repo_id=repo, folder_path=str(folder), path_in_repo=destination)
    remote = {f.path: f for f in api.list_repo_tree(repo, path_in_repo=destination, recursive=True)}
    for path in folder.rglob('*'):
        if not path.is_file() or '.cache' in path.parts or path.name.startswith('.') or path.name == 'CACHEDIR.TAG':
            continue
        key = destination + '/' + path.relative_to(folder).as_posix()
        entry = remote.get(key)
        if entry is None or entry.size != path.stat().st_size:
            raise RuntimeError(f'Artifact size verification failed: {key}')
        with path.open('rb') as handle:
            if getattr(entry, 'lfs', None):
                valid = entry.lfs.sha256 == hashlib.file_digest(handle, 'sha256').hexdigest()
            else:
                blob = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
                for chunk in iter(lambda: handle.read(1024*1024), b''):
                    blob.update(chunk)
                valid = entry.blob_id == blob.hexdigest()
        if not valid:
            raise RuntimeError(f'Artifact hash verification failed: {key}')
