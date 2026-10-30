# intel target

The OAAX Intel runtime, which runs the model through OpenVINO on the host CPU. The converted model
is a zip of OpenVINO IR. The target feeds it the exported model's own input: float32, NCHW, pixels
divided by 255. Detections are judged against the reference detections at an IoU of 0.9.

`deploy-ultralytics fetch intel` downloads the runtime and the toolchain image named in
`target.json`. The runtime archive holds the OpenVINO and TBB libraries next to
`libRuntimeLibrary.so`, and `fetch` extracts all of them.

The Intel toolchain takes a bundle, not the bare exported model. `convert --target intel` puts the
exported model and the files of `bundle/` in a zip and runs the toolchain image on it. The
`config.json` in `bundle/` sets `fp16_compression` to `false`. Without it, the toolchain converts
the model to FP16 and the runtime expects FP16 input.

## Walkthrough

Once per machine, fetch the runtime and the toolchain, and point `detect` at the runtime (`fetch`
prints the exact line):

```sh
deploy-ultralytics fetch intel
export DEPLOY_ULTRALYTICS_RUNTIME=$PWD/runtimes/intel/libRuntimeLibrary.so
```

Then three commands take a model from Ultralytics to boxes on an image:

```sh
deploy-ultralytics export yolov8n.pt
deploy-ultralytics convert yolov8n.onnx --target intel
deploy-ultralytics detect --target intel --model yolov8n.intel.zip fixtures/test-image.jpg
```
