# cpu target

The OAAX reference runtime on the host CPU. The converted model is an ONNX file; the target feeds
it the exported model's own input: float32, NCHW, pixels divided by 255. Detections are judged
against the reference detections at an IoU of 0.9.

`deploy-ultralytics fetch cpu` downloads the runtime and the toolchain image named in
`target.json`; `convert --target cpu` runs that toolchain image, which simplifies the exported ONNX
model.

Before OAAX 2.0 is published, build the runtime from the `oaax-v2` branch of `reference-implementation`
(`runtime-library/build-runtimes.sh X86_64`) and pass the resulting `libRuntimeLibrary.so` to
`detect` with `--runtime` or `DEPLOY_ULTRALYTICS_RUNTIME`.
