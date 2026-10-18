# cpu target

The OAAX reference runtime on the host CPU. The converted model is an ONNX file; the target feeds
it the exported model's own input: float32, NCHW, pixels divided by 255. Detections are judged
against the reference detections at an IoU of 0.9.

`deploy-ultralytics fetch cpu` downloads the runtime and the toolchain image named in
`target.json`; `convert --target cpu` runs that toolchain image, which simplifies the exported ONNX
model.
