# Measurement snapshot

The saved benchmark and judge files were measured on the original corpus with SHA-256 `696a87e561405c090c012875231d68142291cba6908159cc1a6b86af8246e1ea`.

The corpus was subsequently repaired to decode UTF-8 correctly, preserve highlighted code lines, and separate individual API definitions with precise citation anchors. Its current checksum is in `data/corpus_manifest.json`. The previous metrics and cited chunk IDs remain historical evidence; they are not measurements of the current corpus. Evaluation questions and the runtime index were rebuilt. Re-run `python -m eval.run` to obtain fresh metrics before making performance or quality claims about this version. The separate dev_calibration.json artifact, when present, records a development-only gate recommendation for its explicitly recorded corpus hash; it does not establish test accuracy.
