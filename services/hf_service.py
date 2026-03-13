"""
services/hf_service.py
───────────────────────
Reusable Hugging Face import/export services.
Provides programmatic access (HFService class, export_to_hf, import_from_hf)
and a CLI interface for standalone usage. All parameters can be passed directly
or fall back to .env values if omitted.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from dotenv import load_dotenv
from huggingface_hub import HfApi, login, snapshot_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

# ── Logging ────────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)


# ── Result dataclasses ─────────────────────────────────────────────────────────
@dataclass
class ExportResult:
    """Returned by HFService.export() / export_to_hf()"""

    repo_id: str
    repo_path: str
    local_dir: str
    uploaded: List[str] = field(default_factory=list)  # HF paths uploaded

    @property
    def count(self) -> int:
        return len(self.uploaded)

    def __repr__(self) -> str:
        return f"ExportResult(repo='{self.repo_id}/{self.repo_path}', " f"uploaded={self.count} file(s))"


@dataclass
class ImportResult:
    """Returned by HFService.import_data() / import_from_hf()"""

    repo_id: str
    repo_path: str
    downloaded_to: str  # absolute local path
    files: List[str] = field(default_factory=list)  # relative paths

    @property
    def count(self) -> int:
        return len(self.files)

    def __repr__(self) -> str:
        return (
            f"ImportResult(repo='{self.repo_id}/{self.repo_path}', "
            f"downloaded_to='{self.downloaded_to}', files={self.count})"
        )


@dataclass
class RepoInfo:
    """Returned by HFService.info()"""

    id: str
    repo_type: str
    private: bool
    downloads: int
    last_modified: str
    files: List[str] = field(default_factory=list)


# ── Internal helpers ───────────────────────────────────────────────────────────
def _load_env_defaults() -> dict:
    """Load default values from .env (if present). Never raises."""
    load_dotenv()
    return {
        "token": os.getenv("HF_TOKEN", ""),
        "repo_id": os.getenv("HF_REPO_ID", ""),
        "repo_type": os.getenv("HF_REPO_TYPE", "dataset"),
        "data_path": os.getenv("HF_DATA_PATH", "data/train"),
        "local_dir": os.getenv("LOCAL_DATA_DIR", "./local_data"),
        "revision": os.getenv("HF_REVISION", "main"),
    }


# ── Service class ──────────────────────────────────────────────────────────────
class HFService:
    """
    Reusable Hugging Face import/export services.

    Constructor parameters all fall back to .env values when omitted,
    so you can mix explicit args with .env for secrets you'd rather not
    hard-code.

    Parameters
    ----------
    token     : HF access token  (fallback: HF_TOKEN in .env)
    repo_id   : 'user/repo-name' (fallback: HF_REPO_ID in .env)
    repo_type : 'dataset' | 'model'          (fallback: HF_REPO_TYPE, default 'dataset')
    revision  : branch or tag                (fallback: HF_REVISION, default 'main')
    data_path : default repo sub-path        (fallback: HF_DATA_PATH, default 'data/train')
    local_dir : default local directory      (fallback: LOCAL_DATA_DIR, default './local_data')
    """

    def __init__(
        self,
        token: Optional[str] = None,
        repo_id: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        data_path: Optional[str] = None,
        local_dir: Optional[str] = None,
    ):
        env = _load_env_defaults()

        self.token = token or env["token"]
        self.repo_id = repo_id or env["repo_id"]
        self.repo_type = repo_type or env["repo_type"] or "dataset"
        self.revision = revision or env["revision"] or "main"
        self.data_path = data_path or env["data_path"] or "data/train"
        self.local_dir = local_dir or env["local_dir"] or "./local_data"

        if not self.token:
            raise ValueError("HF token is required. Pass token= or set HF_TOKEN in .env")

        self.api = HfApi(token=self.token)
        login(token=self.token, add_to_git_credential=False)
        log.info("HFService authenticated (default repo='%s').", self.repo_id)

    # ── INFO ──────────────────────────────────────────────────────────────────
    def info(
        self,
        repo_id: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> RepoInfo:
        """
        Return metadata about a remote repository.

        Parameters
        ----------
        repo_id   : override instance default
        repo_type : 'dataset' | 'model'
        revision  : branch / tag

        Returns
        -------
        RepoInfo

        Raises
        ------
        RepositoryNotFoundError
        ValueError if repo_id is empty
        """
        rid = repo_id or self.repo_id
        rtyp = repo_type or self.repo_type
        rev = revision or self.revision

        if not rid:
            raise ValueError("repo_id is required for info()")

        try:
            meta = (
                self.api.dataset_info(rid, revision=rev)
                if rtyp == "dataset"
                else self.api.model_info(rid, revision=rev)
            )
        except RepositoryNotFoundError:
            raise RepositoryNotFoundError(f"Repository '{rid}' not found or no access.")

        siblings = getattr(meta, "siblings", []) or []
        return RepoInfo(
            id=meta.id,
            repo_type=rtyp,
            private=meta.private,
            downloads=getattr(meta, "downloads", 0) or 0,
            last_modified=str(meta.last_modified),
            files=[s.rfilename for s in siblings],
        )

    # ── EXPORT (local → HF) ───────────────────────────────────────────────────
    def export(
        self,
        local_dir: Optional[str] = None,
        repo_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        repo_type: Optional[str] = None,
        commit_message: Optional[str] = None,
        private: bool = True,
        file_filter: Optional[Callable[[Path], bool]] = None,
    ) -> ExportResult:
        """
        Upload files from a local directory to the HF repository.

        Parameters
        ----------
        local_dir      : local folder to upload
                         overrides instance default / LOCAL_DATA_DIR in .env
        repo_path      : destination path inside the repo, e.g. 'data/train'
                         overrides instance default / HF_DATA_PATH in .env
        repo_id        : HF repo id, e.g. 'user/reddit-images'
                         overrides instance default / HF_REPO_ID in .env
        repo_type      : 'dataset' | 'model'
        commit_message : custom git commit message (default: 'Upload <path>')
        private        : create repo as private if it doesn't exist yet
        file_filter    : optional callable(Path) -> bool
                         return True to include a file, False to skip it
                         example: file_filter=lambda p: p.suffix == '.parquet'

        Returns
        -------
        ExportResult

        Raises
        ------
        FileNotFoundError  if local_dir does not exist
        ValueError         if repo_id is empty
        """
        ldir = Path(local_dir or self.local_dir)
        rpath = repo_path or self.data_path
        rid = repo_id or self.repo_id
        rtyp = repo_type or self.repo_type

        if not rid:
            raise ValueError("repo_id is required. Pass repo_id= or set HF_REPO_ID in .env")
        if not ldir.exists():
            raise FileNotFoundError(f"Local directory '{ldir}' does not exist.")

        files = [f for f in ldir.rglob("*") if f.is_file()]
        if file_filter:
            files = [f for f in files if file_filter(f)]

        if not files:
            log.warning("No files found in '%s'. Nothing to upload.", ldir)
            return ExportResult(repo_id=rid, repo_path=rpath, local_dir=str(ldir))

        log.info(
            "Exporting %d file(s) from '%s' → %s/%s",
            len(files),
            ldir,
            rid,
            rpath,
        )

        # Ensure repo exists — create if missing
        try:
            self.api.repo_info(rid, repo_type=rtyp)
        except RepositoryNotFoundError:
            log.info("Repository '%s' not found — creating (private=%s).", rid, private)
            self.api.create_repo(rid, repo_type=rtyp, private=private, exist_ok=True)

        uploaded: List[str] = []
        for local_path in files:
            rel = local_path.relative_to(ldir)
            hf_path = f"{rpath}/{rel}".replace("\\", "/")
            msg = commit_message or f"Upload {hf_path}"

            log.info("  ↑ %s → %s", rel, hf_path)
            self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=hf_path,
                repo_id=rid,
                repo_type=rtyp,
                commit_message=msg,
            )
            uploaded.append(hf_path)

        result = ExportResult(
            repo_id=rid,
            repo_path=rpath,
            local_dir=str(ldir),
            uploaded=uploaded,
        )
        log.info("✅ Export complete — %s", result)
        return result

    # ── IMPORT (HF → local) ───────────────────────────────────────────────────
    def import_data(
        self,
        local_dir: Optional[str] = None,
        repo_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
    ) -> ImportResult:
        """
        Download files from the HF repository into a local directory.

        Parameters
        ----------
        local_dir        : destination folder
                           overrides instance default / LOCAL_DATA_DIR in .env
        repo_path        : sub-path inside the repo to pull, e.g. 'data/train'
                           overrides instance default / HF_DATA_PATH in .env
        repo_id          : HF repo id
                           overrides instance default / HF_REPO_ID in .env
        repo_type        : 'dataset' | 'model'
        revision         : branch or tag (default: 'main')
        allow_patterns   : explicit glob list — overrides the repo_path filter
                           e.g. ['data/train/*.parquet', 'data/train/*.json']
        ignore_patterns  : glob list of files to skip
                           e.g. ['*.md', '.gitattributes']

        Returns
        -------
        ImportResult

        Raises
        ------
        RepositoryNotFoundError  if repo doesn't exist / no access
        EntryNotFoundError       if repo_path not found in the repo
        ValueError               if repo_id is empty
        """
        ldir = Path(local_dir or self.local_dir)
        rpath = repo_path or self.data_path
        rid = repo_id or self.repo_id
        rtyp = repo_type or self.repo_type
        rev = revision or self.revision

        if not rid:
            raise ValueError("repo_id is required. Pass repo_id= or set HF_REPO_ID in .env")

        ldir.mkdir(parents=True, exist_ok=True)

        patterns = allow_patterns or [f"{rpath}/*"]
        log.info(
            "Importing %s/%s → '%s' (patterns=%s, revision=%s)",
            rid,
            rpath,
            ldir,
            patterns,
            rev,
        )

        try:
            downloaded_path = snapshot_download(
                repo_id=rid,
                repo_type=rtyp,
                revision=rev,
                allow_patterns=patterns,
                ignore_patterns=ignore_patterns,
                local_dir=str(ldir),
                token=self.token,
            )
        except RepositoryNotFoundError:
            raise RepositoryNotFoundError(f"Repository '{rid}' not found or no access.")
        except EntryNotFoundError:
            raise EntryNotFoundError(f"Path '{rpath}' not found in repo '{rid}'.")

        # snapshot_download preserves the full repo structure, e.g.:
        #   local_dir/data/train/shard-00000.parquet
        # We move files up so they land directly in local_dir:
        #   local_dir/shard-00000.parquet
        nested_dir = Path(downloaded_path) / rpath
        if nested_dir.exists() and nested_dir.is_dir():
            log.info("Flattening '%s' → '%s'", nested_dir, ldir)
            for file in nested_dir.rglob("*"):
                if file.is_file():
                    dest = ldir / file.relative_to(nested_dir)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(file), str(dest))
            # Remove leftover empty repo_path dirs (e.g. data/train, data)
            top_nested = ldir / Path(rpath).parts[0]
            if top_nested.exists():
                shutil.rmtree(str(top_nested))

        base = ldir
        downloaded_files = [str(f.relative_to(base)) for f in base.rglob("*") if f.is_file()]

        result = ImportResult(
            repo_id=rid,
            repo_path=rpath,
            downloaded_to=str(base.resolve()),
            files=downloaded_files,
        )
        log.info("✅ Import complete — %s", result)
        return result


# ── Standalone helper functions ────────────────────────────────────────────────
def export_to_hf(
    token: str,
    repo_id: str,
    local_dir: str,
    repo_path: str = "data/train",
    repo_type: str = "dataset",
    commit_message: Optional[str] = None,
    private: bool = True,
    file_filter: Optional[Callable[[Path], bool]] = None,
) -> ExportResult:
    """
    One-shot export — no need to manage an HFService instance.
    """
    svc = HFService(token=token, repo_id=repo_id, repo_type=repo_type)
    return svc.export(
        local_dir=local_dir,
        repo_path=repo_path,
        commit_message=commit_message,
        private=private,
        file_filter=file_filter,
    )


def import_from_hf(
    token: str,
    repo_id: str,
    local_dir: str,
    repo_path: str = "data/train",
    repo_type: str = "dataset",
    revision: str = "main",
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
) -> ImportResult:
    """
    One-shot import — no need to manage an HFService instance.
    """
    svc = HFService(token=token, repo_id=repo_id, repo_type=repo_type, revision=revision)
    return svc.import_data(
        local_dir=local_dir,
        repo_path=repo_path,
        revision=revision,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hugging Face repo import / export services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--token", help="HuggingFace access token (overrides HF_TOKEN in .env)")
    shared.add_argument("--repo", help="Override HF_REPO_ID")
    shared.add_argument("--path", help="Override HF_DATA_PATH  (e.g. data/train)")
    shared.add_argument("--local-dir", dest="local_dir", help="Override LOCAL_DATA_DIR")
    shared.add_argument("--repo-type", dest="repo_type", help="dataset | model")

    exp = sub.add_parser("export", parents=[shared], help="Upload local files → HF repo")
    exp.add_argument("--commit-msg", dest="commit_msg", help="Custom commit message")
    exp.add_argument("--public", action="store_true", help="Create repo as public")

    imp = sub.add_parser("import", parents=[shared], help="Download HF repo → local files")
    imp.add_argument("--revision", help="Override HF_REVISION (branch/tag)")
    imp.add_argument("--allow-patterns", dest="allow_patterns", nargs="*", help="Glob patterns to include")
    imp.add_argument("--ignore-patterns", dest="ignore_patterns", nargs="*", help="Glob patterns to exclude")

    inf = sub.add_parser("info", parents=[shared], help="Print repo metadata")
    inf.add_argument("--revision", help="Branch / tag")

    return parser


def _cli_main():
    parser = _build_parser()
    args = parser.parse_args()
    env = _load_env_defaults()

    try:
        svc = HFService(
            token=getattr(args, "token", None) or env.get("token"),
            repo_id=getattr(args, "repo", None) or env.get("repo_id"),
            repo_type=getattr(args, "repo_type", None) or env.get("repo_type"),
        )

        if args.command == "export":
            result = svc.export(
                local_dir=args.local_dir,
                repo_path=args.path,
                commit_message=args.commit_msg,
                private=not args.public,
            )
            print(result)

        elif args.command == "import":
            result = svc.import_data(
                local_dir=args.local_dir,
                repo_path=args.path,
                revision=args.revision,
                allow_patterns=args.allow_patterns,
                ignore_patterns=args.ignore_patterns,
            )
            print(result)

        elif args.command == "info":
            info = svc.info(revision=getattr(args, "revision", None))
            print(f"\n── Repository info ─────────────────────────────")
            print(f"  ID        : {info.id}")
            print(f"  Type      : {info.repo_type}")
            print(f"  Private   : {info.private}")
            print(f"  Downloads : {info.downloads}")
            print(f"  Last mod  : {info.last_modified}")
            if info.files:
                print(f"\n  Files ({len(info.files)}):")
                for f in info.files:
                    print(f"    {f}")
            print()

    except (ValueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        sys.exit(1)
    except RepositoryNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
