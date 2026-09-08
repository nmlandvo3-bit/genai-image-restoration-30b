from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent

PIPELINE = (
    ROOT
    / "scripts"
    / "final_multispecialist_restoration_pipeline_30b.py"
)

RESULTS_ROOT = (
    ROOT
    / "gui_results_30b"
)

SUPPORTED_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


CLASS_LABELS = {
    "clean": "Clean / No damage",
    "noise": "Noise",
    "blur": "Blur",
    "scratch": "Scratch",
    "missing": "Missing region",
    "irregular": "Irregular missing region",
}


SPECIALIST_LABELS = {
    "preserve": "Preserve original",

    "denoise":
        "30B fine-tuned Restormer denoiser",

    "deblur":
        "Fine-tuned Restormer deblur",

    "scratch_inpaint":
        "Navier-Stokes inpainting (r7)",

    "inpaint":
        "Telea inpainting (r15)",
}


class RestorationGUI(tk.Tk):

    def __init__(self):

        super().__init__()

        self.title(
            "30B Multi-Specialist Image Restoration"
        )

        self.geometry(
            "1450x900"
        )

        self.minsize(
            1100,
            720,
        )

        self.input_path: Path | None = None
        self.output_path: Path | None = None
        self.mask_path: Path | None = None
        self.report_path: Path | None = None
        self.current_run_dir: Path | None = None

        self.original_photo = None
        self.restored_photo = None
        self.mask_photo = None

        self.processing = False

        self.process: (
            subprocess.Popen
            | None
        ) = None

        self.started_at = None

        self._configure_style()
        self._build_ui()
        self._check_project()


    # =========================================================================
    # STYLE
    # =========================================================================

    def _configure_style(self):

        style = ttk.Style(
            self
        )

        try:

            style.theme_use(
                "clam"
            )

        except tk.TclError:

            pass

        style.configure(
            "Title.TLabel",
            font=(
                "Segoe UI",
                20,
                "bold",
            ),
        )

        style.configure(
            "Heading.TLabel",
            font=(
                "Segoe UI",
                11,
                "bold",
            ),
        )

        style.configure(
            "Result.TLabel",
            font=(
                "Segoe UI",
                10,
            ),
        )

        style.configure(
            "Primary.TButton",
            font=(
                "Segoe UI",
                10,
                "bold",
            ),
            padding=(
                16,
                9,
            ),
        )

        style.configure(
            "Secondary.TButton",
            font=(
                "Segoe UI",
                10,
            ),
            padding=(
                12,
                8,
            ),
        )


    # =========================================================================
    # BUILD UI
    # =========================================================================

    def _build_ui(self):

        self.columnconfigure(
            0,
            weight=1,
        )

        self.rowconfigure(
            1,
            weight=1,
        )

        # ---------------------------------------------------------------------
        # HEADER
        # ---------------------------------------------------------------------

        header = ttk.Frame(
            self,
            padding=(
                18,
                14,
            ),
        )

        header.grid(
            row=0,
            column=0,
            sticky="ew",
        )

        header.columnconfigure(
            1,
            weight=1,
        )

        ttk.Label(
            header,
            text=(
                "30B Image Restoration"
            ),
            style="Title.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        self.project_status_var = (
            tk.StringVar(
                value=(
                    "Checking project..."
                )
            )
        )

        ttk.Label(
            header,
            textvariable=(
                self.project_status_var
            ),
        ).grid(
            row=0,
            column=1,
            sticky="e",
        )

        # ---------------------------------------------------------------------
        # IMAGE PANELS
        # ---------------------------------------------------------------------

        main = ttk.Frame(
            self,
            padding=(
                18,
                0,
                18,
                12,
            ),
        )

        main.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        for col in range(
            3
        ):

            main.columnconfigure(
                col,
                weight=1,
            )

        main.rowconfigure(
            0,
            weight=1,
        )

        self.original_panel = (
            self._make_image_panel(
                main,
                "Original",
                0,
            )
        )

        self.restored_panel = (
            self._make_image_panel(
                main,
                "Restored",
                1,
            )
        )

        self.mask_panel = (
            self._make_image_panel(
                main,
                "Predicted Damage Mask",
                2,
            )
        )

        # ---------------------------------------------------------------------
        # BOTTOM AREA
        # ---------------------------------------------------------------------

        bottom = ttk.Frame(
            self,
            padding=(
                18,
                0,
                18,
                18,
            ),
        )

        bottom.grid(
            row=2,
            column=0,
            sticky="ew",
        )

        bottom.columnconfigure(
            1,
            weight=1,
        )

        # ---------------------------------------------------------------------
        # CONTROLS
        # ---------------------------------------------------------------------

        controls = ttk.Frame(
            bottom
        )

        controls.grid(
            row=0,
            column=0,
            sticky="nw",
            padx=(
                0,
                20,
            ),
        )

        self.select_button = (
            ttk.Button(
                controls,
                text="Select Image",
                style=(
                    "Secondary.TButton"
                ),
                command=(
                    self.select_image
                ),
            )
        )

        self.select_button.grid(
            row=0,
            column=0,
            padx=(
                0,
                8,
            ),
        )

        self.restore_button = (
            ttk.Button(
                controls,
                text="Restore Image",
                style=(
                    "Primary.TButton"
                ),
                command=(
                    self.start_restoration
                ),
                state="disabled",
            )
        )

        self.restore_button.grid(
            row=0,
            column=1,
            padx=(
                0,
                8,
            ),
        )

        self.open_output_button = (
            ttk.Button(
                controls,
                text=(
                    "Open Output Folder"
                ),
                style=(
                    "Secondary.TButton"
                ),
                command=(
                    self.open_output_folder
                ),
                state="disabled",
            )
        )

        self.open_output_button.grid(
            row=0,
            column=2,
            padx=(
                0,
                8,
            ),
        )

        self.save_copy_button = (
            ttk.Button(
                controls,
                text=(
                    "Save Restored Copy"
                ),
                style=(
                    "Secondary.TButton"
                ),
                command=(
                    self.save_restored_copy
                ),
                state="disabled",
            )
        )

        self.save_copy_button.grid(
            row=0,
            column=3,
        )

        # ---------------------------------------------------------------------
        # RESULTS
        # ---------------------------------------------------------------------

        results_frame = (
            ttk.LabelFrame(
                bottom,
                text=(
                    "Restoration Result"
                ),
                padding=12,
            )
        )

        results_frame.grid(
            row=0,
            column=1,
            sticky="nsew",
        )

        for col in range(
            8
        ):

            results_frame.columnconfigure(
                col,
                weight=1,
            )

        self.damage_var = (
            tk.StringVar(
                value="—"
            )
        )

        self.confidence_var = (
            tk.StringVar(
                value="—"
            )
        )

        self.specialist_var = (
            tk.StringVar(
                value="—"
            )
        )

        self.mask_fraction_var = (
            tk.StringVar(
                value="—"
            )
        )

        self.detector_var = (
            tk.StringVar(
                value="—"
            )
        )

        self.size_var = (
            tk.StringVar(
                value="—"
            )
        )

        self.elapsed_var = (
            tk.StringVar(
                value="—"
            )
        )

        self.status_var = (
            tk.StringVar(
                value="Ready"
            )
        )

        fields = [
            (
                "Detected damage",
                self.damage_var,
            ),
            (
                "Confidence",
                self.confidence_var,
            ),
            (
                "Specialist",
                self.specialist_var,
            ),
            (
                "Damage area",
                self.mask_fraction_var,
            ),
            (
                "Detector route",
                self.detector_var,
            ),
            (
                "Image size",
                self.size_var,
            ),
            (
                "Processing time",
                self.elapsed_var,
            ),
            (
                "Status",
                self.status_var,
            ),
        ]

        for index, (
            label_text,
            variable,
        ) in enumerate(
            fields
        ):

            row = (
                index
                // 4
            )

            col = (
                index
                % 4
            ) * 2

            ttk.Label(
                results_frame,
                text=label_text,
                style=(
                    "Heading.TLabel"
                ),
            ).grid(
                row=row,
                column=col,
                sticky="w",
                padx=(
                    0,
                    6,
                ),
                pady=3,
            )

            ttk.Label(
                results_frame,
                textvariable=variable,
                style=(
                    "Result.TLabel"
                ),
            ).grid(
                row=row,
                column=col + 1,
                sticky="w",
                padx=(
                    0,
                    14,
                ),
                pady=3,
            )

        # ---------------------------------------------------------------------
        # PROGRESS
        # ---------------------------------------------------------------------

        progress_frame = (
            ttk.Frame(
                self,
                padding=(
                    18,
                    0,
                    18,
                    16,
                ),
            )
        )

        progress_frame.grid(
            row=3,
            column=0,
            sticky="ew",
        )

        progress_frame.columnconfigure(
            0,
            weight=1,
        )

        self.progress = (
            ttk.Progressbar(
                progress_frame,
                mode="indeterminate",
            )
        )

        self.progress.grid(
            row=0,
            column=0,
            sticky="ew",
        )

        self.progress_text_var = (
            tk.StringVar(
                value=(
                    "Select an image "
                    "to begin."
                )
            )
        )

        ttk.Label(
            progress_frame,
            textvariable=(
                self.progress_text_var
            ),
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(
                6,
                0,
            ),
        )


    # =========================================================================
    # IMAGE PANEL
    # =========================================================================

    def _make_image_panel(
        self,
        parent,
        title: str,
        column: int,
    ):

        frame = ttk.LabelFrame(
            parent,
            text=title,
            padding=10,
        )

        frame.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=(
                0
                if column == 0
                else 6,

                0
                if column == 2
                else 6,
            ),
        )

        frame.columnconfigure(
            0,
            weight=1,
        )

        frame.rowconfigure(
            0,
            weight=1,
        )

        label = tk.Label(
            frame,
            text="No image",
            bg="#1e1e1e",
            fg="#d0d0d0",
            anchor="center",
            relief="flat",
        )

        label.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        return label


    # =========================================================================
    # PROJECT CHECK
    # =========================================================================

    def _check_project(self):

        if not PIPELINE.exists():

            self.project_status_var.set(
                "30B pipeline not found"
            )

            self.restore_button.config(
                state="disabled"
            )

            messagebox.showerror(
                "30B Pipeline Not Found",
                (
                    "Place this GUI file in "
                    "the root of your "
                    "genai-image-restoration "
                    "project.\n\n"
                    f"Expected pipeline:\n"
                    f"{PIPELINE}"
                ),
            )

            return

        self.project_status_var.set(
            "30B pipeline ready"
        )


    # =========================================================================
    # SELECT IMAGE
    # =========================================================================

    def select_image(self):

        if self.processing:
            return

        selected = (
            filedialog.askopenfilename(
                title=(
                    "Select image to restore"
                ),
                filetypes=[
                    (
                        "Image files",
                        "*.png *.jpg *.jpeg "
                        "*.bmp *.tif *.tiff "
                        "*.webp",
                    ),
                    (
                        "All files",
                        "*.*",
                    ),
                ],
            )
        )

        if not selected:
            return

        path = Path(
            selected
        )

        if (
            path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):

            messagebox.showerror(
                "Unsupported File",
                (
                    "Please select a "
                    "supported image file."
                ),
            )

            return

        try:

            with Image.open(
                path
            ) as image:

                image.verify()

        except Exception as exc:

            messagebox.showerror(
                "Invalid Image",
                (
                    "Could not open this "
                    f"image:\n\n{exc}"
                ),
            )

            return

        self.input_path = (
            path
        )

        self._reset_result_state()

        self._show_image(
            path,
            self.original_panel,
            "original",
        )

        self.restore_button.config(
            state="normal"
        )

        self.progress_text_var.set(
            f"Selected: {path.name}"
        )


    # =========================================================================
    # START RESTORATION
    # =========================================================================

    def start_restoration(self):

        if (
            self.processing
            or self.input_path
            is None
        ):

            return

        timestamp = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        stem = (
            self.input_path.stem
        )

        self.current_run_dir = (
            RESULTS_ROOT
            / f"{timestamp}_{stem}"
        )

        self.current_run_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        self.output_path = (
            self.current_run_dir
            / f"{stem}_restored.png"
        )

        self.mask_path = (
            self.current_run_dir
            / f"{stem}_mask.png"
        )

        self.report_path = (
            self.current_run_dir
            / f"{stem}_report.json"
        )

        self.processing = True

        self.started_at = (
            time.perf_counter()
        )

        self.restore_button.config(
            state="disabled"
        )

        self.select_button.config(
            state="disabled"
        )

        self.open_output_button.config(
            state="disabled"
        )

        self.save_copy_button.config(
            state="disabled"
        )

        self.status_var.set(
            "Processing"
        )

        self.progress_text_var.set(
            (
                "Running detector "
                "and classifier..."
            )
        )

        self.progress.start(
            10
        )

        threading.Thread(
            target=(
                self._run_pipeline_worker
            ),
            daemon=True,
        ).start()


    # =========================================================================
    # RUN PIPELINE
    # =========================================================================

    def _run_pipeline_worker(self):

        command = [
            sys.executable,
            str(PIPELINE),
            "--input",
            str(self.input_path),
            "--output",
            str(self.output_path),
            "--mask-output",
            str(self.mask_path),
            "--report",
            str(self.report_path),
        ]

        try:

            self.process = (
                subprocess.Popen(
                    command,
                    cwd=str(ROOT),
                    stdout=(
                        subprocess.PIPE
                    ),
                    stderr=(
                        subprocess.STDOUT
                    ),
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            )

            captured = []

            assert (
                self.process.stdout
                is not None
            )

            for line in (
                self.process.stdout
            ):

                captured.append(
                    line
                )

                text = (
                    line.strip()
                )

                if text:

                    self.after(
                        0,
                        self._update_progress_from_pipeline,
                        text,
                    )

            return_code = (
                self.process.wait()
            )

            console_output = (
                "".join(
                    captured
                )
            )

            self.after(
                0,
                self._pipeline_finished,
                return_code,
                console_output,
            )

        except Exception as exc:

            self.after(
                0,
                self._pipeline_exception,
                exc,
            )


    # =========================================================================
    # PROGRESS
    # =========================================================================

    def _update_progress_from_pipeline(
        self,
        text: str,
    ):

        lowered = (
            text.lower()
        )

        if (
            "running damage detector"
            in lowered
        ):

            message = (
                "Detecting damaged regions..."
            )

        elif (
            "running damage classifier"
            in lowered
        ):

            message = (
                "Classifying damage type..."
            )

        elif (
            "loading 30b"
            in lowered
            or
            "running 30b noise"
            in lowered
        ):

            message = (
                "Restoring noise with "
                "30B fine-tuned Restormer..."
            )

        elif (
            "deblur inference"
            in lowered
        ):

            message = (
                "Restoring blur with "
                "fine-tuned Restormer..."
            )

        elif (
            "navier-stokes"
            in lowered
        ):

            message = (
                "Repairing scratches..."
            )

        elif (
            "telea"
            in lowered
        ):

            message = (
                "Repairing missing region..."
            )

        elif (
            "restoration complete"
            in lowered
        ):

            message = (
                "Finalizing restoration..."
            )

        else:

            return

        self.progress_text_var.set(
            message
        )


    # =========================================================================
    # PIPELINE FINISHED
    # =========================================================================

    def _pipeline_finished(
        self,
        return_code: int,
        console_output: str,
    ):

        elapsed = (
            time.perf_counter()
            - self.started_at

            if self.started_at
            is not None

            else 0.0
        )

        self.progress.stop()

        self.processing = False

        self.process = None

        self.select_button.config(
            state="normal"
        )

        self.restore_button.config(
            state=(
                "normal"
                if self.input_path
                is not None
                else "disabled"
            )
        )

        if return_code != 0:

            self.status_var.set(
                "Failed"
            )

            self.progress_text_var.set(
                "Restoration failed."
            )

            error_log = (
                self.current_run_dir
                / "pipeline_error.txt"
            )

            error_log.write_text(
                console_output,
                encoding="utf-8",
            )

            messagebox.showerror(
                "Restoration Failed",
                (
                    "The 30B pipeline "
                    "returned an error.\n\n"
                    f"Log saved to:\n"
                    f"{error_log}"
                ),
            )

            return

        required = [
            self.output_path,
            self.mask_path,
            self.report_path,
        ]

        if not all(
            path is not None
            and path.exists()
            for path in required
        ):

            self.status_var.set(
                "Failed"
            )

            messagebox.showerror(
                "Missing Output",
                (
                    "30B completed but one "
                    "or more output files "
                    "were not created."
                ),
            )

            return

        try:

            report = json.loads(
                self.report_path.read_text(
                    encoding="utf-8"
                )
            )

            self._populate_report(
                report,
                elapsed,
            )

            self._show_image(
                self.output_path,
                self.restored_panel,
                "restored",
            )

            self._show_image(
                self.mask_path,
                self.mask_panel,
                "mask",
            )

            self.open_output_button.config(
                state="normal"
            )

            self.save_copy_button.config(
                state="normal"
            )

            self.progress_text_var.set(
                "Restoration complete."
            )

            self.status_var.set(
                "Complete"
            )

        except Exception as exc:

            self.status_var.set(
                "Report error"
            )

            messagebox.showerror(
                "Result Error",
                (
                    "Restoration completed, "
                    "but the GUI could not "
                    "load the result:\n\n"
                    f"{exc}"
                ),
            )


    # =========================================================================
    # PIPELINE EXCEPTION
    # =========================================================================

    def _pipeline_exception(
        self,
        exc: Exception,
    ):

        self.progress.stop()

        self.processing = False

        self.process = None

        self.select_button.config(
            state="normal"
        )

        self.restore_button.config(
            state=(
                "normal"
                if self.input_path
                is not None
                else "disabled"
            )
        )

        self.status_var.set(
            "Failed"
        )

        self.progress_text_var.set(
            (
                "Could not start "
                "the pipeline."
            )
        )

        messagebox.showerror(
            "Pipeline Error",
            str(exc),
        )


    # =========================================================================
    # REPORT
    # =========================================================================

    def _populate_report(
        self,
        report: dict,
        elapsed: float,
    ):

        detector = (
            report.get(
                "detector",
                {},
            )
        )

        classifier = (
            report.get(
                "classifier",
                {},
            )
        )

        routing = (
            report.get(
                "routing",
                {},
            )
        )

        predicted_class = (
            classifier.get(
                "predicted_class",
                "unknown",
            )
        )

        confidence = (
            classifier.get(
                "confidence",
                0.0,
            )
        )

        specialist = (
            routing.get(
                "specialist",
                "unknown",
            )
        )

        fraction = (
            detector.get(
                "predicted_mask_fraction",
                0.0,
            )
        )

        routed_damage = (
            detector.get(
                "routed_as_damage",
                False,
            )
        )

        width = (
            report.get(
                "image_width",
                "?",
            )
        )

        height = (
            report.get(
                "image_height",
                "?",
            )
        )

        self.damage_var.set(
            CLASS_LABELS.get(
                predicted_class,
                predicted_class,
            )
        )

        self.confidence_var.set(
            f"{float(confidence) * 100:.2f}%"
        )

        self.specialist_var.set(
            SPECIALIST_LABELS.get(
                specialist,
                specialist,
            )
        )

        self.mask_fraction_var.set(
            f"{float(fraction) * 100:.3f}%"
        )

        self.detector_var.set(
            (
                "Damage"
                if routed_damage
                else "Preserve"
            )
        )

        self.size_var.set(
            f"{width} × {height}"
        )

        self.elapsed_var.set(
            f"{elapsed:.1f} s"
        )


    # =========================================================================
    # IMAGE DISPLAY
    # =========================================================================

    def _show_image(
        self,
        path: Path,
        label: tk.Label,
        slot: str,
    ):

        with Image.open(
            path
        ) as image:

            image = image.convert(
                "RGB"
            )

            image.thumbnail(
                (
                    430,
                    560,
                ),
                Image.Resampling.LANCZOS,
            )

            photo = (
                ImageTk.PhotoImage(
                    image
                )
            )

        label.config(
            image=photo,
            text="",
        )

        if slot == "original":

            self.original_photo = (
                photo
            )

        elif slot == "restored":

            self.restored_photo = (
                photo
            )

        elif slot == "mask":

            self.mask_photo = (
                photo
            )


    # =========================================================================
    # OPEN OUTPUT
    # =========================================================================

    def open_output_folder(self):

        if (
            self.current_run_dir
            is None
            or not
            self.current_run_dir.exists()
        ):

            return

        try:

            if sys.platform.startswith(
                "win"
            ):

                import os

                os.startfile(
                    self.current_run_dir
                )

            else:

                messagebox.showinfo(
                    "Output Folder",
                    str(
                        self.current_run_dir
                    ),
                )

        except Exception as exc:

            messagebox.showerror(
                "Open Folder Failed",
                str(exc),
            )


    # =========================================================================
    # SAVE COPY
    # =========================================================================

    def save_restored_copy(self):

        if (
            self.output_path
            is None
            or not
            self.output_path.exists()
        ):

            return

        destination = (
            filedialog.asksaveasfilename(
                title=(
                    "Save restored image"
                ),
                defaultextension=".png",
                initialfile=(
                    self.output_path.name
                ),
                filetypes=[
                    (
                        "PNG image",
                        "*.png",
                    ),
                    (
                        "JPEG image",
                        "*.jpg *.jpeg",
                    ),
                    (
                        "All files",
                        "*.*",
                    ),
                ],
            )
        )

        if not destination:
            return

        destination_path = Path(
            destination
        )

        try:

            if (
                destination_path
                .suffix.lower()
                in (
                    ".jpg",
                    ".jpeg",
                )
            ):

                with Image.open(
                    self.output_path
                ) as image:

                    image.convert(
                        "RGB"
                    ).save(
                        destination_path,
                        quality=95,
                    )

            else:

                import shutil

                shutil.copy2(
                    self.output_path,
                    destination_path,
                )

            messagebox.showinfo(
                "Saved",
                (
                    "Restored image "
                    f"saved to:\n"
                    f"{destination_path}"
                ),
            )

        except Exception as exc:

            messagebox.showerror(
                "Save Failed",
                str(exc),
            )


    # =========================================================================
    # RESET
    # =========================================================================

    def _reset_result_state(self):

        self.output_path = None
        self.mask_path = None
        self.report_path = None
        self.current_run_dir = None

        self.restored_panel.config(
            image="",
            text=(
                "No restored image"
            ),
        )

        self.mask_panel.config(
            image="",
            text="No mask",
        )

        self.restored_photo = None
        self.mask_photo = None

        self.damage_var.set(
            "—"
        )

        self.confidence_var.set(
            "—"
        )

        self.specialist_var.set(
            "—"
        )

        self.mask_fraction_var.set(
            "—"
        )

        self.detector_var.set(
            "—"
        )

        self.size_var.set(
            "—"
        )

        self.elapsed_var.set(
            "—"
        )

        self.status_var.set(
            "Ready"
        )

        self.open_output_button.config(
            state="disabled"
        )

        self.save_copy_button.config(
            state="disabled"
        )


# =============================================================================
# START
# =============================================================================

if __name__ == "__main__":

    app = RestorationGUI()

    app.mainloop()