"""Shared engine for building paired CT-slice segmentation datasets."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .naming import extract_nodule_id, extract_slice_index, sample_filename
from .spec import DatasetSpec
from .validation import load_mask, load_volume, validate_paired_shapes


@dataclass(frozen=True)
class StudyPlan:
    """Validated source files required to process one study."""

    study_dir: Path
    volume_paths: dict[str, Path]
    nodule_masks: tuple[tuple[Path, tuple[Path, ...]], ...]


@dataclass(frozen=True)
class BuildSummary:
    """Result and counters from one dataset build."""

    metadata: pd.DataFrame
    studies: int
    samples: int
    created: int
    skipped: int


def require_directory(path: str | Path, description: str) -> Path:
    """Return an existing directory or raise a descriptive error."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected {description} directory: {path}")
    return path


def create_build_plan(spec: DatasetSpec) -> list[StudyPlan]:
    """Validate source structure before writing any prepared samples."""
    mask_root = require_directory(
        spec.consensus_mask_dir,
        f"{spec.display_name} consensus mask input",
    )
    for name, directory in spec.ct_volume_dirs.items():
        require_directory(directory, f"{spec.display_name} {name} input")

    study_dirs = sorted(path for path in mask_root.iterdir() if path.is_dir())
    if not study_dirs:
        raise FileNotFoundError(f"No study directories found in: {mask_root}")

    plans: list[StudyPlan] = []
    for study_dir in study_dirs:
        volume_paths = {
            name: directory / f"{study_dir.name}.npy"
            for name, directory in spec.ct_volume_dirs.items()
        }
        missing_volumes = [
            path for path in volume_paths.values() if not path.is_file()
        ]
        if missing_volumes:
            raise FileNotFoundError(
                f"Missing CT volumes for {study_dir.name}: {missing_volumes}"
            )

        nodule_dirs = sorted(path for path in study_dir.iterdir() if path.is_dir())
        if not nodule_dirs:
            raise FileNotFoundError(f"No nodule directories found in: {study_dir}")

        nodule_masks: list[tuple[Path, tuple[Path, ...]]] = []
        for nodule_dir in nodule_dirs:
            extract_nodule_id(nodule_dir, spec.nodule_prefix)
            mask_paths = tuple(
                sorted(
                    nodule_dir.glob("slice_*.npy"),
                    key=extract_slice_index,
                )
            )
            if not mask_paths:
                raise FileNotFoundError(
                    f"No consensus mask slices found in: {nodule_dir}"
                )
            nodule_masks.append((nodule_dir, mask_paths))

        plans.append(
            StudyPlan(
                study_dir=study_dir,
                volume_paths=volume_paths,
                nodule_masks=tuple(nodule_masks),
            )
        )

    return plans


def save_paired_sample(
    ct_slices: dict[str, np.ndarray],
    mask: np.ndarray,
    ct_output_paths: dict[str, Path],
    mask_output_path: Path,
    overwrite: bool,
) -> str:
    """Save one complete sample or skip an already complete output pair."""
    output_paths = [*ct_output_paths.values(), mask_output_path]
    existing = [path.is_file() for path in output_paths]
    if any(existing) and not overwrite:
        if all(existing):
            return "skipped"
        present = [path for path, exists in zip(output_paths, existing) if exists]
        missing = [path for path, exists in zip(output_paths, existing) if not exists]
        raise FileExistsError(
            f"Incomplete prepared sample; present={present}, missing={missing}. "
            "Use overwrite=True to rebuild the pair."
        )

    for name, output_path in ct_output_paths.items():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, ct_slices[name])

    mask_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mask_output_path, mask.astype(bool, copy=False))
    return "created"


def process_study(
    spec: DatasetSpec,
    plan: StudyPlan,
    overwrite: bool = False,
) -> tuple[list[dict[str, object]], int, int]:
    """Create every prepared sample for one validated study plan."""
    volumes = {
        name: load_volume(path)
        for name, path in plan.volume_paths.items()
    }
    study_name = plan.study_dir.name
    records: list[dict[str, object]] = []
    created = 0
    skipped = 0

    for nodule_dir, mask_paths in plan.nodule_masks:
        nodule_id = extract_nodule_id(nodule_dir, spec.nodule_prefix)
        for mask_path in mask_paths:
            slice_index = extract_slice_index(mask_path)
            ct_slices: dict[str, np.ndarray] = {}
            for name, volume in volumes.items():
                if slice_index >= volume.shape[0]:
                    raise IndexError(
                        f"{study_name} slice {slice_index} is outside {name} "
                        f"volume depth {volume.shape[0]}."
                    )
                ct_slices[name] = np.asarray(volume[slice_index])

            mask = load_mask(mask_path)
            validate_paired_shapes(ct_slices, mask, str(mask_path))
            filename = sample_filename(
                study_name,
                nodule_dir.name,
                slice_index,
            )
            ct_output_paths = {
                name: spec.ct_output_dirs[name] / filename
                for name in spec.ct_volume_dirs
            }
            mask_output_path = spec.mask_output_dir / filename
            status = save_paired_sample(
                ct_slices=ct_slices,
                mask=mask,
                ct_output_paths=ct_output_paths,
                mask_output_path=mask_output_path,
                overwrite=overwrite,
            )
            created += status == "created"
            skipped += status == "skipped"

            record: dict[str, object] = {
                "filename": filename,
                spec.study_column: study_name,
                spec.identifier_column: nodule_id,
                "slice_index": slice_index,
            }
            for name, ct_slice in ct_slices.items():
                record[f"{name}_path"] = ct_output_paths[name].relative_to(
                    spec.output_dir
                ).as_posix()
                record[f"{name}_height"] = ct_slice.shape[0]
                record[f"{name}_width"] = ct_slice.shape[1]
            record.update(
                {
                    "mask_path": mask_output_path.relative_to(
                        spec.output_dir
                    ).as_posix(),
                    "mask_height": mask.shape[0],
                    "mask_width": mask.shape[1],
                    "mask_pixels": int(mask.sum()),
                }
            )
            records.append(record)

    return records, created, skipped


def build_dataset(
    spec: DatasetSpec,
    overwrite: bool = False,
    plans: list[StudyPlan] | None = None,
) -> BuildSummary:
    """Build one complete source-specific segmentation dataset."""
    if spec.metadata_csv.exists() and not overwrite:
        raise FileExistsError(
            f"Output metadata already exists: {spec.metadata_csv}. "
            "Use overwrite=True or --overwrite to replace it."
        )

    plans = create_build_plan(spec) if plans is None else plans
    records: list[dict[str, object]] = []
    created = 0
    skipped = 0
    for plan in tqdm(
        plans,
        desc=f"Building {spec.display_name} dataset",
        unit="study",
    ):
        study_records, study_created, study_skipped = process_study(
            spec,
            plan,
            overwrite=overwrite,
        )
        records.extend(study_records)
        created += study_created
        skipped += study_skipped

    metadata = pd.DataFrame(records)
    if metadata.empty:
        raise RuntimeError(f"No {spec.display_name} samples were generated.")
    metadata = metadata.sort_values(
        [spec.study_column, spec.identifier_column, "slice_index"],
        kind="stable",
    ).reset_index(drop=True)
    spec.metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(spec.metadata_csv, index=False)

    return BuildSummary(
        metadata=metadata,
        studies=len(plans),
        samples=len(metadata),
        created=created,
        skipped=skipped,
    )
