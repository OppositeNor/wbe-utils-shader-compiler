# Copyright 2026 OppositeNor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from .shader_metadata import ShaderMetadataNormalizer


# Defined for the second, metadata-only compile, so metadata-only declarations reach the reflection.
REFLECTION_METADATA_MACRO = "WBE_REFLECTION_METADATA"


class SlangCompiler:
    """
    Compiles Slang sources to validated SPIR-V and normalized shader metadata with the slangc and spirv-val on PATH.

    Each compile runs slangc twice. The first run writes the SPIR-V and the reflection; the second, with
    WBE_REFLECTION_METADATA=1 defined, reflects the declarations only that build sees into the same metadata file, and its
    SPIR-V is validated and discarded. The metadata is then normalized with ShaderMetadataNormalizer. The compiler holds
    only its configuration, so it may compile concurrently into distinct outputs.
    """

    def __init__(self, profile: str, include_dirs: list[Path], debug_info: bool = False) -> None:
        """
        :param profile: The target profile, optionally followed by `+`-separated capabilities, as slangc's -profile.
        :param include_dirs: Include directories, in search order.
        :param debug_info: Whether to emit full debug information (slangc -g3).
        """
        self._profile = profile
        self._include_dirs = list(include_dirs)
        self._debug_info = debug_info
        self._normalizer = ShaderMetadataNormalizer()

    def compile(self, source_path: Path, output_path: Path, metadata_path: Path, stage: str | None, entry: str | None,
            macro_definitions: list[tuple[str, str]]) -> None:
        """
        Compile one source.

        :param source_path: The Slang source.
        :param output_path: Receives the SPIR-V.
        :param metadata_path: Receives the normalized metadata.
        :param stage: The stage of entry, or None with entry None to compile every [shader(...)] entry point.
        :param entry: The entry point, or None with stage None.
        :param macro_definitions: The (name, value) macro definitions to preprocess with.
        :raises RuntimeError: If only one of stage and entry is given, slangc or spirv-val is not on PATH, or compiling,
            reflecting, or validating fails; the message holds the tool's output.
        """
        if (stage is None) != (entry is None):
            raise RuntimeError(f"Shader {source_path.as_posix()} must declare both stage and entry, or neither.")
        slangc_path = shutil.which("slangc")
        if slangc_path is None:
            raise RuntimeError("Failed to find valid slang compiler.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        command = [slangc_path, "-target", "spirv", "-profile", self._profile]
        if entry is not None and stage is not None:
            command.extend(["-entry", entry, "-stage", stage])
        for name, value in macro_definitions:
            command.append(f"-D{name}={value}")
        for include_dir in self._include_dirs:
            command.extend(["-I", str(include_dir)])
        if self._debug_info:
            command.append("-g3")
        self._run_slangc([*command, "-o", str(output_path), "-reflection-json", str(metadata_path), str(source_path)],
            f"Failed to compile shader {source_path.as_posix()}")
        self._validate_spirv(source_path, output_path)
        reflection_output_path = output_path.with_name(f"{output_path.name}.reflection")
        try:
            self._run_slangc([*command, f"-D{REFLECTION_METADATA_MACRO}=1", "-o", str(reflection_output_path),
                "-reflection-json", str(metadata_path), str(source_path)],
                f"Failed to reflect shader {source_path.as_posix()}")
            self._validate_spirv(source_path, reflection_output_path)
        finally:
            reflection_output_path.unlink(missing_ok=True)
        raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_path.write_text(json.dumps(self._normalizer.normalize(raw_metadata), indent=4), encoding="utf-8")

    def _run_slangc(self, command: list[str], message: str) -> None:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"{message}:\n{result.stderr}")

    def _validate_spirv(self, source_path: Path, spirv_path: Path) -> None:
        spirv_val_path = shutil.which("spirv-val")
        if spirv_val_path is None:
            raise RuntimeError("Failed to find spirv-val required for shader validation.")
        result = subprocess.run([spirv_val_path, str(spirv_path), "--relax-block-layout", "--target-env", "vulkan1.3"],
            capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to validate the compiled SPIR-V of shader {source_path.as_posix()}:\n{result.stdout}{result.stderr}")
