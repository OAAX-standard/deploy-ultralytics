# cpu target

The OAAX reference runtime on the host CPU. The converted model is an ONNX file; the target feeds
it the exported model's own input: float32, NCHW, pixels divided by 255. Detections are judged
against the reference detections at an IoU of 0.9.

`deploy-ultralytics fetch cpu` downloads the runtime and the toolchain image named in
`target.json`; `convert --target cpu` runs that toolchain image, which simplifies the exported ONNX
model.

## Walkthrough

Once per machine, fetch the runtime and the toolchain, and point `detect` at the runtime (`fetch`
prints the exact line):

```sh
deploy-ultralytics fetch cpu
export DEPLOY_ULTRALYTICS_RUNTIME=$PWD/runtimes/cpu/libRuntimeLibrary.so
```

Then three commands take a model from Ultralytics to boxes on an image:

```sh
deploy-ultralytics export yolov8n.pt
deploy-ultralytics convert yolov8n.onnx --target cpu
deploy-ultralytics detect --target cpu --model yolov8n.cpu.onnx fixtures/test-image.jpg
```
