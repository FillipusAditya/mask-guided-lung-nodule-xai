"""Dataset-specific paths and naming rules for sample generation."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class DatasetSpec:
    """Describe one source dataset consumed by the shared builder."""

    slug: str
    display_name: str
    ct_volume_dirs: Mapping[str, Path]
    consensus_mask_dir: Path
    output_dir: Path
    nodule_prefix: str
    identifier_column: str
    study_column: str

    def __post_init__(self) -> None:
        """Normalize paths and reject incomplete specifications."""
        if not self.slug or not self.display_name:
            raise ValueError("Dataset slug and display name must be nonempty.")
        if not self.ct_volume_dirs:
            raise ValueError("At least one CT representation is required.")
        if not self.nodule_prefix or not self.identifier_column:
            raise ValueError("Nodule prefix and identifier column are required.")

        normalized = {
            str(name): Path(path)
            for name, path in self.ct_volume_dirs.items()
        }
        object.__setattr__(
            self,
            "ct_volume_dirs",
            MappingProxyType(normalized),
        )
        object.__setattr__(self, "consensus_mask_dir", Path(self.consensus_mask_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    @property
    def ct_output_dirs(self) -> dict[str, Path]:
        """Return output directories for every CT representation."""
        return {
            name: self.output_dir / name
            for name in self.ct_volume_dirs
        }

    @property
    def mask_output_dir(self) -> Path:
        """Return the prepared mask output directory."""
        return self.output_dir / "mask"

    @property
    def metadata_csv(self) -> Path:
        """Return the prepared sample metadata path."""
        return self.output_dir / "metadata.csv"

